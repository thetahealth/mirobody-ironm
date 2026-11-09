"""问答语音输入的接口（spec [2026-10-23] §199）：网页录好一段 WAV 传上来，转成文字回给它填进输入框。

录音只在内存里转、不落盘；文字不记日志，用户点发送后才和打字的消息一样存。
同一个人同时只转一段；整台服务器同时最多转两段，免得把 CPU 占满。
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Annotated, Any

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

from ironm_agent import asr

_log = logging.getLogger("ironm_agent.voice")
router: APIRouter = APIRouter()
_busy: set[str] = set()
_slots = asyncio.Semaphore(2)


def _verify() -> Any:
    """和 web._verify 一样（mirobody 的登录校验）；不从 web 导入，免得互相导入。"""
    from mirobody.server.auth import verify_token

    return verify_token


AUDIO_TYPES = {"audio/wav", "audio/x-wav", "audio/wave", "audio/vnd.wave"}


@router.get("/iron/api/voice")
async def voice_status(user_id: str = Depends(_verify())) -> dict[str, Any]:
    """装好了语音模型才显示话筒。"""
    return {"available": asr.available(), "max_seconds": asr.MAX_SECONDS}


@router.post("/iron/api/voice/transcribe")
async def transcribe(
    audio: Annotated[UploadFile | None, File()] = None, user_id: str = Depends(_verify())
) -> dict[str, Any]:
    if not asr.available():
        raise HTTPException(503, "语音输入还没装好，先打字吧")
    if audio is None:
        raise HTTPException(422, "没收到录音")
    if (audio.content_type or "").split(";")[0].strip().lower() not in AUDIO_TYPES:
        raise HTTPException(415, "录音格式不对")
    data = await audio.read(asr.MAX_BYTES + 1)
    if len(data) > asr.MAX_BYTES:
        raise HTTPException(413, "录音太大了，一段最长 60 秒")
    if user_id in _busy:
        raise HTTPException(429, "上一段还在转，稍等一下")
    _busy.add(user_id)
    started = time.monotonic()
    try:
        async with _slots:
            text = await asyncio.to_thread(asr.transcribe, data)
    except asr.NotHeard as e:
        raise HTTPException(422, str(e)) from None
    except asr.AsrError as e:
        raise HTTPException(400, str(e)) from None
    finally:
        _busy.discard(user_id)
        del data
    # 只记时长和耗时，不记文字
    _log.info("voice transcribed: ms=%d chars=%d", (time.monotonic() - started) * 1000, len(text))
    return {"text": text}
