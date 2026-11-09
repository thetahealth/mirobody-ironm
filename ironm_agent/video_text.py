"""视频转文字（spec [2026-10-23] §200，视频出方案的第一步）：取出声音，按段转写，按句标出自第几秒。

- 取声音用 PyAV（自带 FFmpeg 库，BSD / LGPL，见 web/THIRD-PARTY.md），转成 16kHz 单声道；
  转写用 §199 的同一个语音模型（asr.py），健身词按同一张纠错表纠。
- 长录音按段转：每段 10–25 秒，在这段里最安静的地方切开，免得把一句话切成两半；
  模型给出每个字的时间，按句号、问号、叹号分句，每句记它第一个字的时间。
- 这里只管算，不碰文件以外的东西：存、排队、进度在 web_video.py。
"""

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from ironm_agent import asr

RATE = 16000
MAX_SECONDS = 30 * 60  # 最长 30 分钟
MAX_BYTES = 500 * 1024 * 1024  # 最大 500 MB
SEG_MIN, SEG_MAX = 10.0, 25.0  # 每段 10–25 秒
FRAME = 0.03  # 找安静处时按 30 毫秒一帧算响度
SILENT = 0.01  # 一段最响处不到满幅 1%：当没人说话，不转
ENDS = "。？！?!；;"
LONG_SENTENCE = 60  # 一句超过这么多字还没句号的，在逗号处断开，免得一条太长
PAUSE = 0.8  # 两个字之间停了这么久：模型常不打句号，这里也当一句结束


class VideoError(ValueError):
    """视频本身的问题（打不开、没有声音、太长）：说给用户听的话。"""


class NoSpeech(VideoError):
    """转出来没有字：纯音乐、没人说话。"""


def probe_seconds(path: Path) -> float:
    """不解码，只读时长（打不开、没有声音的当场拒）。"""
    import av

    try:
        with av.open(str(path)) as c:
            if not c.streams.audio:
                raise VideoError("这个视频里没有声音")
            s = c.streams.audio[0]
            if c.duration:
                return c.duration / 1_000_000
            if s.duration and s.time_base:
                return float(s.duration * s.time_base)
            return 0.0
    except VideoError:
        raise
    except Exception as e:  # av 对坏文件、不是视频的文件抛各种错
        raise VideoError("这个文件打不开，换一个视频或声音文件试试") from e


def decode_audio(path: Path) -> Any:
    """整段声音 → 16kHz 单声道 float32（-1..1）。超过 30 分钟的停下并拒。"""
    import av
    import numpy as np

    parts: list[Any] = []
    total = 0
    limit = (MAX_SECONDS + 5) * RATE
    try:
        with av.open(str(path)) as c:
            if not c.streams.audio:
                raise VideoError("这个视频里没有声音")
            stream = c.streams.audio[0]
            resampler = av.AudioResampler(format="flt", layout="mono", rate=RATE)
            for frame in c.decode(stream):
                for out in resampler.resample(frame):
                    a = out.to_ndarray().reshape(-1)
                    parts.append(a)
                    total += len(a)
                if total > limit:
                    raise VideoError("视频太长了，最长 30 分钟")
            for out in resampler.resample(None):
                parts.append(out.to_ndarray().reshape(-1))
    except VideoError:
        raise
    except Exception as e:
        raise VideoError("这个文件的声音取不出来，换一个视频或声音文件试试") from e
    if not parts:
        raise NoSpeech("这个视频里没听到人说话")
    return np.concatenate(parts).astype(np.float32)


def cut_points(samples: Any) -> list[tuple[int, int]]:
    """切成 10–25 秒一段，每次在 10–25 秒之间最安静的那一帧切。"""
    import numpy as np

    n = len(samples)
    hop = int(FRAME * RATE)
    if n == 0:
        return []
    frames = n // hop
    loud = (
        np.sqrt((samples[: frames * hop].reshape(frames, hop).astype(np.float64) ** 2).mean(axis=1))
        if frames
        else np.zeros(0)
    )
    out: list[tuple[int, int]] = []
    start = 0
    while start < n:
        if n - start <= SEG_MAX * RATE:
            out.append((start, n))
            break
        lo = -(-(start + int(SEG_MIN * RATE)) // hop)  # 向上取整：切点不早于 10 秒
        hi = (start + int(SEG_MAX * RATE)) // hop
        f = lo + int(np.argmin(loud[lo:hi]))
        end = f * hop + hop // 2
        out.append((start, end))
        start = end
    return out


def _sentences(tokens: list[str], times: list[float], offset: float) -> list[dict[str, Any]]:
    """一段里的字和时间 → [{t: 第几秒, text: 一句}]。"""
    out: list[dict[str, Any]] = []
    buf: list[str] = []
    first = 0.0

    def flush() -> None:
        text = asr.fix_words("".join(buf).strip()).rstrip("，,、")
        if text and text[-1] not in ENDS:
            text += "。"
        if text.strip("，。？！、,.?!；; "):
            out.append({"t": round(offset + first, 1), "text": text})
        buf.clear()

    last = -1.0
    for tok, t in zip(tokens, times, strict=False):
        if buf and last >= 0 and t - last >= PAUSE:
            flush()
        last = t
        if not buf:
            if tok.strip() == "" or tok in "，,、":
                continue
            first = t
        buf.append(tok)
        if tok in ENDS or (tok in "，," and len("".join(buf)) >= LONG_SENTENCE):
            flush()
    if buf:
        flush()
    return out


def transcribe_samples(
    samples: Any, progress: Callable[[float], None] | None = None
) -> list[dict[str, Any]]:
    """整段声音 → 按句的文字和时间。没转出字的抛 NoSpeech。"""
    rec = asr._get_recognizer()
    segs = cut_points(samples)
    out: list[dict[str, Any]] = []
    for i, (a, b) in enumerate(segs):
        piece = samples[a:b]
        if len(piece) and float(abs(piece).max()) >= SILENT:
            stream = rec.create_stream()
            stream.accept_waveform(RATE, piece)
            rec.decode_stream(stream)
            r = stream.result
            out += _sentences(list(r.tokens), list(r.timestamps), a / RATE)
        if progress:
            progress((i + 1) / len(segs))
    if not out:
        raise NoSpeech("这个视频里没听到人说话")
    return out


def transcribe_file(path: Path, progress: Callable[[str, float], None] | None = None) -> dict[str, Any]:
    """一个视频或声音文件 → {seconds, sentences, ms}。同步、吃 CPU：调用方放到线程里跑。"""
    started = time.monotonic()
    if progress:
        progress("audio", 0)
    samples = decode_audio(path)
    seconds = len(samples) / RATE
    if seconds > MAX_SECONDS + 5:
        raise VideoError("视频太长了，最长 30 分钟")
    sentences = transcribe_samples(samples, (lambda p: progress("text", p)) if progress else None)
    return {
        "seconds": round(seconds),
        "sentences": sentences,
        "ms": round((time.monotonic() - started) * 1000),
    }
