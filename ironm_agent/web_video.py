"""视频转文字的接口（spec [2026-10-23] §200）：方案库里上传视频，后台转成按句带时间的文字，存下来给下一步生成方案用。

- 上传先存到系统临时目录里的一个工作目录，读完时长不对的当场拒；转写在后台跑，整台服务器同时只转一个，
  别人在转时排队。工作目录在转完（成功或失败）后整个删掉：视频、声音都不留。
- 进度放在内存里（服务重启就没了，这时显示"中断了，再传一次"）；转好的文字加密存在 DietStore 的 "videotext" 种类下。
- 文字只是中间稿（§200 需求 5）：长期只留核对过的两份方案；下一步做好之前最多留 7 天，读列表、点开、再传时把过期的删掉。
- 日志只记时长、句数、耗时，不记文件名和文字。
"""

from __future__ import annotations

import asyncio
import logging
import shutil
import tempfile
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Annotated, Any

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

from ironm_agent import asr, video_text
from ironm_agent.web_voice import _verify

_log = logging.getLogger("ironm_agent.video_text")
router: APIRouter = APIRouter()
KIND = "videotext"
PLAN_KIND = "videoplan"  # 按这段文字生成的两份方案（§201），和文字同一个编号，文字删了它也删
LIMIT = 20
KEEP_DAYS = 7  # 转好的文字最多留几天
CHUNK = 1024 * 1024
_jobs: dict[str, dict[str, Any]] = {}  # 用户 → 正在转的这一个（或刚失败、还没看到的）
_tasks: set[asyncio.Task[None]] = set()
_one = asyncio.Lock()  # 整台服务器同时只转一个


def store_dep() -> Any:
    from ironm_agent.recipes import default_diet_store

    return default_diet_store()


STORE = Depends(store_dep)


def _expires(x: dict[str, Any]) -> datetime | None:
    try:
        made = datetime.fromisoformat(str(x.get("created") or ""))
    except ValueError:
        return None
    return (made if made.tzinfo else made.replace(tzinfo=UTC)) + timedelta(days=KEEP_DAYS)


def _days_left(x: dict[str, Any], now: datetime) -> int:
    end = _expires(x)
    return max(0, -(-(end - now) // timedelta(days=1))) if end else 0


async def _alive(store: Any, user_id: str) -> list[dict[str, Any]]:
    """本人还没过期的视频文字；过期的（和读不出时间的）顺手删掉。"""
    now = datetime.now(UTC)
    keep = []
    for x in await store.all(user_id, KIND):
        if not isinstance(x, dict) or not x.get("id"):
            continue
        end = _expires(x)
        if end is None or end <= now:
            await store.delete(user_id, KIND, x["id"])
            await store.delete(user_id, PLAN_KIND, x["id"])
            continue
        keep.append(x)
    return keep


def _summary(x: dict[str, Any], now: datetime | None = None) -> dict[str, Any]:
    return {
        "id": x["id"], "name": x.get("name", ""), "seconds": x.get("seconds", 0),
        "created": x.get("created", ""), "count": len(x.get("sentences") or []),
        "days_left": _days_left(x, now or datetime.now(UTC)),
    }  # fmt: skip


def _clean_name(name: str) -> str:
    stem = Path(name or "").name.rsplit(".", 1)[0].strip()
    return " ".join(stem.split())[:40] or "视频"


async def _run(user_id: str, job: dict[str, Any], work: Path, store: Any) -> None:
    loop = asyncio.get_running_loop()

    def progress(state: str, p: float) -> None:  # 在转写线程里调
        loop.call_soon_threadsafe(job.update, {"state": state, "progress": round(p, 2)})

    try:
        job["state"] = "queued" if _one.locked() else "audio"
        async with _one:
            job["state"] = "audio"
            out = await asyncio.to_thread(video_text.transcribe_file, work / "upload", progress)
        item = {
            "id": job["id"], "name": job["name"], "seconds": out["seconds"],
            "created": datetime.now(UTC).isoformat(timespec="seconds"), "sentences": out["sentences"],
        }  # fmt: skip
        await store.put(user_id, KIND, job["id"], item)
        _jobs.pop(user_id, None)
        _log.info(
            "video text done: seconds=%d sentences=%d ms=%d",
            out["seconds"], len(out["sentences"]), out["ms"],
        )  # fmt: skip
    except video_text.VideoError as e:
        job.update(state="failed", error=str(e))
        _log.info("video text refused: reason=%s", type(e).__name__)
    except Exception as e:
        job.update(state="failed", error="这次没转成，稍后再传一次")
        _log.warning("video text failed: error_type=%s", type(e).__name__)
    finally:
        shutil.rmtree(work, ignore_errors=True)


def _job_view(job: dict[str, Any] | None) -> dict[str, Any] | None:
    if job is None:
        return None
    keys = ("id", "name", "state", "progress", "error")
    return {k: job.get(k) for k in keys if k in job}


@router.get("/iron/api/video-text")
async def list_video_text(user_id: str = Depends(_verify()), store: Any = STORE) -> dict[str, Any]:
    from ironm_agent.web_video_plan import plan_job_view

    rows = sorted(await _alive(store, user_id), key=lambda x: x.get("created", ""), reverse=True)
    planned = {x.get("id") for x in await store.all(user_id, PLAN_KIND) if isinstance(x, dict)}
    job = _jobs.get(user_id)
    if job and job.get("state") == "failed":
        _jobs.pop(user_id, None)  # 失败的说一次就清掉
    return {
        "available": asr.available(),
        "limit": LIMIT,
        "max_mb": video_text.MAX_BYTES // (1024 * 1024),
        "max_minutes": video_text.MAX_SECONDS // 60,
        "keep_days": KEEP_DAYS,
        "job": _job_view(job),
        "items": [{**_summary(x), "planned": x["id"] in planned} for x in rows],
        "plan_job": plan_job_view(user_id),
    }


@router.get("/iron/api/video-text/{item_id}")
async def get_video_text(item_id: str, user_id: str = Depends(_verify()), store: Any = STORE) -> dict[str, Any]:
    x = next((r for r in await _alive(store, user_id) if r["id"] == item_id), None)
    if not x:
        raise HTTPException(404, "没有这条视频文字")
    return {**_summary(x), "sentences": x.get("sentences") or []}


@router.delete("/iron/api/video-text/{item_id}")
async def delete_video_text(item_id: str, user_id: str = Depends(_verify()), store: Any = STORE) -> dict[str, Any]:
    if not await store.delete(user_id, KIND, item_id):
        raise HTTPException(404, "没有这条视频文字")
    await store.delete(user_id, PLAN_KIND, item_id)
    return {"deleted": item_id}


@router.post("/iron/api/video-text")
async def upload_video(
    video: Annotated[UploadFile | None, File()] = None,
    user_id: str = Depends(_verify()),
    store: Any = STORE,
) -> dict[str, Any]:
    if not asr.available():
        raise HTTPException(503, "语音模型还没装好，暂时转不了")
    if video is None or not video.filename:
        raise HTTPException(422, "选一个视频")
    mime = (video.content_type or "").lower()
    if mime and not mime.startswith(("video/", "audio/")) and mime != "application/octet-stream":
        raise HTTPException(415, "只收视频或声音文件")
    if (job := _jobs.get(user_id)) and job.get("state") != "failed":
        raise HTTPException(429, "上一个视频还在转，转完再传")
    if len(await _alive(store, user_id)) >= LIMIT:
        raise HTTPException(409, f"最多存 {LIMIT} 条视频文字，先删掉几条不用的")
    work = Path(tempfile.mkdtemp(prefix="ironm-video-"))
    try:
        size = 0
        with (work / "upload").open("wb") as f:
            while chunk := await video.read(CHUNK):
                size += len(chunk)
                if size > video_text.MAX_BYTES:
                    raise HTTPException(413, "视频太大了，最大 500 MB")
                f.write(chunk)
        if size == 0:
            raise HTTPException(422, "这个文件是空的")
        try:
            seconds = await asyncio.to_thread(video_text.probe_seconds, work / "upload")
        except video_text.VideoError as e:
            raise HTTPException(415, str(e)) from None
        if seconds > video_text.MAX_SECONDS:
            raise HTTPException(413, "视频太长了，最长 30 分钟")
    except BaseException:
        shutil.rmtree(work, ignore_errors=True)
        raise
    job = {"id": uuid.uuid4().hex[:12], "name": _clean_name(video.filename), "state": "queued", "progress": 0}
    _jobs[user_id] = job
    task = asyncio.create_task(_run(user_id, job, work, store))
    _tasks.add(task)
    task.add_done_callback(_tasks.discard)
    return {"job": _job_view(job)}
