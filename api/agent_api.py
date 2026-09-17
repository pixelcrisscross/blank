"""
Minimal ADK event-stream bridge for the ORCA dashboard.

Exposes the existing `app.agent.root_agent` Runner over Server-Sent Events
so the standalone `orca_dashboard.html` can watch the real ORCA pipeline
execute — no workflow duplication, no fake data.
"""
from __future__ import annotations

import asyncio
import json
import uuid
from typing import Any, AsyncGenerator

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from app.agent import root_agent


router = APIRouter(prefix="/agent", tags=["orca-agent"])

APP_NAME = "orca_dashboard"
USER_ID = "dashboard_user"

_session_service = InMemorySessionService()
_runner: Runner | None = None
_runner_lock = asyncio.Lock()


async def _get_runner() -> Runner:
    global _runner
    if _runner is None:
        async with _runner_lock:
            if _runner is None:
                _runner = Runner(
                    app_name=APP_NAME,
                    agent=root_agent,
                    session_service=_session_service,
                )
    return _runner


# ── JSON-safety ────────────────────────────────────────────────────────────

def _json_safe(obj: Any, depth: int = 0) -> Any:
    if depth > 12:
        return "<truncated>"
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, dict):
        return {str(k): _json_safe(v, depth + 1) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_json_safe(v, depth + 1) for v in obj]
    if hasattr(obj, "model_dump"):
        try:
            return _json_safe(obj.model_dump(), depth + 1)
        except Exception:
            pass
    if hasattr(obj, "__dict__"):
        return {
            k: _json_safe(v, depth + 1)
            for k, v in vars(obj).items()
            if not k.startswith("_")
        }
    return str(obj)


def _serialize_event(event: Any) -> dict:
    content = getattr(event, "content", None)
    parts_out: list[dict] = []
    text_full = ""
    if content is not None:
        for part in (getattr(content, "parts", None) or []):
            txt = getattr(part, "text", None)
            if txt:
                text_full += txt
            fc = getattr(part, "function_call", None)
            fr = getattr(part, "function_response", None)
            parts_out.append({
                "text": txt,
                "function_call": _json_safe(fc) if fc else None,
                "function_response": _json_safe(fr) if fr else None,
            })

    try:
        is_final = event.is_final_response()
    except Exception:
        is_final = None

    ts = getattr(event, "timestamp", None)
    usage = getattr(event, "usage_metadata", None)

    return {
        "author": getattr(event, "author", None),
        "invocation_id": getattr(event, "invocation_id", None),
        "is_final_response": is_final,
        "text": text_full,
        "parts": parts_out,
        "timestamp": str(ts) if ts is not None else None,
        "error_code": getattr(event, "error_code", None),
        "error_message": getattr(event, "error_message", None),
        "usage_metadata": _json_safe(usage) if usage else None,
        "actions": _json_safe(getattr(event, "actions", None)),
    }


def _sse(event: str, data: Any) -> str:
    payload = json.dumps(data, default=str)
    return f"event: {event}\ndata: {payload}\n\n"


# ── Routes ─────────────────────────────────────────────────────────────────

class RunRequest(BaseModel):
    prompt: str
    session_id: str | None = None


@router.get("/health")
async def health() -> dict:
    return {
        "status": "ok",
        "service": "orca-agent-bridge",
        "app_name": APP_NAME,
        "root_agent": getattr(root_agent, "name", "unknown"),
    }


@router.post("/session")
async def create_session() -> dict:
    sid = f"dash_{uuid.uuid4().hex[:12]}"
    await _session_service.create_session(
        app_name=APP_NAME, user_id=USER_ID, session_id=sid,
    )
    return {"session_id": sid, "app_name": APP_NAME, "user_id": USER_ID}


@router.get("/session/{session_id}")
async def get_session(session_id: str) -> dict:
    try:
        session = await _session_service.get_session(
            app_name=APP_NAME, user_id=USER_ID, session_id=session_id,
        )
    except Exception as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    state = getattr(session, "state", {}) or {}
    return {
        "session_id": session_id,
        "app_name": APP_NAME,
        "user_id": USER_ID,
        "state": _json_safe(state),
    }


@router.post("/run")
async def run_agent(req: RunRequest) -> StreamingResponse:
    if not req.prompt or not req.prompt.strip():
        raise HTTPException(status_code=400, detail="Empty prompt")

    session_id = req.session_id or f"dash_{uuid.uuid4().hex[:12]}"

    try:
        existing = await _session_service.get_session(
            app_name=APP_NAME, user_id=USER_ID, session_id=session_id,
        )
    except Exception:
        existing = None
    if existing is None:
        await _session_service.create_session(
            app_name=APP_NAME, user_id=USER_ID, session_id=session_id,
        )

    async def stream() -> AsyncGenerator[str, None]:
        yield _sse("start", {
            "session_id": session_id,
            "prompt": req.prompt,
            "app_name": APP_NAME,
            "user_id": USER_ID,
            "root_agent": getattr(root_agent, "name", "unknown"),
        })
        seq = 0
        try:
            runner = await _get_runner()
            content = types.Content(
                role="user", parts=[types.Part(text=req.prompt)],
            )
            async for event in runner.run_async(
                user_id=USER_ID,
                session_id=session_id,
                new_message=content,
            ):
                seq += 1
                payload = _serialize_event(event)
                payload["seq"] = seq
                yield _sse("event", payload)
        except Exception as exc:
            yield _sse("error", {
                "error": str(exc),
                "type": type(exc).__name__,
            })
        finally:
            yield _sse("end", {"session_id": session_id, "event_count": seq})

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )