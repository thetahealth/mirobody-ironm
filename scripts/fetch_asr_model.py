"""下载问答语音输入用的语音模型（spec [2026-10-23] §199）到本机数据目录。

用法：.venv/bin/python scripts/fetch_asr_model.py
- 模型是 SenseVoice-Small 原版的 int8 版（sherpa-onnx 转好的，2024-07-17）。别换成 2025-09-09 那一版：
  那是粤语微调版（WSYue-ASR），普通话不出标点、还会当成粤语。
- 压缩包约 163 MB，解开后只留 model.int8.onnx、tokens.txt 和说明，放在 <数据目录>/asr/sense-voice（约 230 MB）。
- 已经有了就什么都不做；下到一半断了，下次从头再下。下载完核对 SHA-256，对不上就不装。
- 下载慢时可以用 IRONM_ASR_MODEL_URL 换一个地址（内容要和默认的一样，否则核对不过）。
- 模型许可：FunASR 模型许可（免费可商用，要注明出处、保留模型名），见 ironm_agent/web/THIRD-PARTY.md。
"""

from __future__ import annotations

import hashlib
import os
import shutil
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NAME = "sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17"
URL = f"https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/{NAME}.tar.bz2"
SHA256 = "7d1efa2138a65b0b488df37f8b89e3d91a60676e416f515b952358d83dfd347e"
KEEP = {"model.int8.onnx", "tokens.txt", "LICENSE", "README.md"}
UA = "iron-m/0.1 (self-hosted fitness app; model download)"


def target() -> Path:
    explicit = os.environ.get("IRONM_ASR_MODEL_DIR")
    if explicit:
        return Path(explicit)
    return Path(os.environ.get("IRONM_DATA_DIR") or ROOT / "data") / "asr" / "sense-voice"


def main() -> int:
    out = target()
    if (out / "model.int8.onnx").is_file() and (out / "tokens.txt").is_file():
        print(f"语音模型已经在 {out}")
        return 0
    out.parent.mkdir(parents=True, exist_ok=True)
    url = os.environ.get("IRONM_ASR_MODEL_URL") or URL
    with tempfile.TemporaryDirectory(dir=out.parent, prefix=".asr-") as tmp:
        archive = Path(tmp) / "model.tar.bz2"
        digest = hashlib.sha256()
        req = urllib.request.Request(url, headers={"User-Agent": UA})
        with urllib.request.urlopen(req, timeout=60) as resp, archive.open("wb") as f:
            while chunk := resp.read(1 << 20):
                digest.update(chunk)
                f.write(chunk)
        if digest.hexdigest() != SHA256:
            print("语音模型下载下来核对不上，没装；可以再跑一次", file=sys.stderr)
            return 1
        staged = Path(tmp) / "sense-voice"
        staged.mkdir()
        with tarfile.open(archive, "r:bz2") as tar:
            for m in tar.getmembers():
                name = Path(m.name).name
                if m.isfile() and name in KEEP and Path(m.name).parent.name == NAME:
                    src = tar.extractfile(m)
                    if src is not None:
                        with (staged / name).open("wb") as dst:
                            shutil.copyfileobj(src, dst)
        if not (staged / "model.int8.onnx").is_file() or not (staged / "tokens.txt").is_file():
            print("语音模型压缩包里缺文件，没装", file=sys.stderr)
            return 1
        if out.exists():
            shutil.rmtree(out)
        staged.rename(out)
    print(f"语音模型装好了：{out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
