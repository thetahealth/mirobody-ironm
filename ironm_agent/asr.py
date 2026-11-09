"""问答语音输入（spec [2026-10-23] §199）：服务器上自己装的语音模型把一段录音转成文字。

- 模型：SenseVoice-Small（FunASR 模型许可，署名见 web/THIRD-PARTY.md），用 sherpa-onnx 跑 int8 版。
  放在数据目录 asr/sense-voice 下（model.int8.onnx、tokens.txt），由 scripts/fetch_asr_model.py 下载；
  也可以用 IRONM_ASR_MODEL_DIR 指到别处。没装好时 available() 为 False，网页不显示话筒。
- 录音只在内存里转，不写盘、不进日志；转出的文字也不记日志（发出去之后才和打字的消息一样存）。
- 健身词转错的，按 asr_fixes.yaml 的纠错表改。
"""

from __future__ import annotations

import io
import logging
import os
import re
import threading
import wave
from functools import cache
from pathlib import Path
from typing import Any

from ironm_agent.yamlfile import load_yaml

_log = logging.getLogger("ironm_agent.asr")
ROOT = Path(__file__).resolve().parent.parent
FIXES_FILE = Path(__file__).with_name("asr_fixes.yaml")

MAX_SECONDS = 60  # 一段最长 60 秒（网页到时自动停）；服务端多留 1 秒余量
MIN_SECONDS = 0.5  # 不到半秒的当没说
SILENT_PEAK = 0.01  # 整段最响的地方都不到满幅的 1%：当没听到说话
MAX_BYTES = 6 * 1024 * 1024  # 48kHz 双声道 60 秒约 11MB；网页会先转成 16kHz 单声道（约 2MB）


class AsrError(ValueError):
    """录音本身有问题（格式不对、太长）：说给用户听的话。"""


class NotHeard(AsrError):
    """太短、太轻或者没转出字：请他再说一次。"""


def model_dir() -> Path:
    explicit = os.environ.get("IRONM_ASR_MODEL_DIR")
    if explicit:
        return Path(explicit)
    return Path(os.environ.get("IRONM_DATA_DIR") or ROOT / "data") / "asr" / "sense-voice"


def _model_files(d: Path) -> tuple[Path, Path] | None:
    model = next((d / n for n in ("model.int8.onnx", "model.onnx") if (d / n).is_file()), None)
    tokens = d / "tokens.txt"
    return (model, tokens) if model and tokens.is_file() else None


def available() -> bool:
    """模型文件在、运行库装了：网页才显示话筒。"""
    if _model_files(model_dir()) is None:
        return False
    try:
        import sherpa_onnx  # noqa: F401
    except ImportError:
        return False
    return True


_lock = threading.Lock()
_recognizer: Any = None
_loaded_from: Path | None = None


def _get_recognizer() -> Any:
    """第一次用时载入（约 1–2 秒），之后一直用这一个。"""
    global _recognizer, _loaded_from
    d = model_dir()
    with _lock:
        if _recognizer is not None and _loaded_from == d:
            return _recognizer
        files = _model_files(d)
        if files is None:
            raise RuntimeError("语音模型没装")
        import sherpa_onnx

        threads = max(1, min(4, (os.cpu_count() or 2) // 2))
        _recognizer = sherpa_onnx.OfflineRecognizer.from_sense_voice(
            model=str(files[0]), tokens=str(files[1]), num_threads=threads, use_itn=True, language="auto"
        )
        _loaded_from = d
        return _recognizer


def read_wav(data: bytes) -> tuple[int, Any]:
    """网页传来的 WAV（16 位 PCM，单声道或多声道）→ (采样率, -1..1 的单声道浮点数组)。"""
    import numpy as np

    if len(data) > MAX_BYTES:
        raise AsrError("录音太大了，一段最长 60 秒")
    try:
        with wave.open(io.BytesIO(data)) as w:
            rate, channels, width, frames = w.getframerate(), w.getnchannels(), w.getsampwidth(), w.getnframes()
            raw = w.readframes(frames)
    except (wave.Error, EOFError) as e:
        raise AsrError("录音格式不对") from e
    if width != 2 or not 8000 <= rate <= 48000 or channels < 1:
        raise AsrError("录音格式不对")
    samples = np.frombuffer(raw[: len(raw) // (2 * channels) * 2 * channels], dtype="<i2").astype(np.float32)
    if channels > 1:
        samples = samples.reshape(-1, channels).mean(axis=1)
    samples /= 32768.0
    seconds = len(samples) / rate
    if seconds > MAX_SECONDS + 1:
        raise AsrError("录音太长了，一段最长 60 秒")
    if seconds < MIN_SECONDS or (len(samples) and float(abs(samples).max()) < SILENT_PEAK) or not len(samples):
        raise NotHeard("没听清，再说一次")
    return rate, samples


@cache
def _fixes() -> tuple[tuple[str, str], ...]:
    """纠错表：长的先换，免得短的把长的拆坏。"""
    raw = load_yaml(FIXES_FILE) or {}
    pairs = [(str(k), str(v)) for k, v in (raw.get("fixes") or {}).items() if k and k != v]
    return tuple(sorted(pairs, key=lambda kv: -len(kv[0])))


# 模型偶尔把"十二次"写成"1二次"：阿拉伯数字后面紧跟一个中文数字、再跟单位的，并成阿拉伯数字（§200 实测）
_CN_DIGIT = {"零": "0", "〇": "0", "一": "1", "二": "2", "三": "3", "四": "4", "五": "5", "六": "6", "七": "7", "八": "8", "九": "9"}
_MIXED = re.compile(r"(\d)([零〇一二三四五六七八九])(?=[次组个下秒分公斤磅千卡克%])")


def fix_words(text: str) -> str:
    for wrong, right in _fixes():
        text = text.replace(wrong, right)
    return _MIXED.sub(lambda m: m.group(1) + _CN_DIGIT[m.group(2)], text)


def transcribe(data: bytes) -> str:
    """一段 WAV → 文字（带标点、纠过健身词）。同步、吃 CPU：调用方放到线程里跑。"""
    rate, samples = read_wav(data)
    rec = _get_recognizer()
    stream = rec.create_stream()
    stream.accept_waveform(rate, samples)
    rec.decode_stream(stream)
    text = fix_words((stream.result.text or "").strip())
    if not text.strip("，。？！、,.?! "):
        raise NotHeard("没听清，再说一次")
    return text
