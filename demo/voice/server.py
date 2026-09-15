"""FastAPI server for the ORCA voice agent."""

from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from fastapi import BackgroundTasks, FastAPI
from fastapi.responses import FileResponse, RedirectResponse
import uvicorn

from pipecat.transports.smallwebrtc.request_handler import (
    SmallWebRTCPatchRequest,
    SmallWebRTCRequest,
    SmallWebRTCRequestHandler,
)

from demo.voice.bot import run_bot


app = FastAPI(title="ORCA Voice")
handler = SmallWebRTCRequestHandler()
CLIENT_FILE = Path(__file__).parent / "client" / "index.html"


@app.post("/api/offer")
async def offer(request: SmallWebRTCRequest, background_tasks: BackgroundTasks):
    async def on_connection(connection):
        background_tasks.add_task(run_bot, connection)

    return await handler.handle_web_request(
        request=request,
        webrtc_connection_callback=on_connection,
    )


@app.patch("/api/offer")
async def ice_candidate(request: SmallWebRTCPatchRequest):
    await handler.handle_patch_request(request)
    return {"status": "success"}


@app.get("/client")
async def serve_client():
    return FileResponse(CLIENT_FILE)


@app.get("/")
async def root():
    return RedirectResponse(url="/client")


@app.get("/health")
async def health():
    return {"status": "ok", "service": "orca-voice"}


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    await handler.close()


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7861, log_level="info")