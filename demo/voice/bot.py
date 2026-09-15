"""
Pipecat voice agent for ORCA.

Pipeline:
    Mic -> Whisper STT -> Ollama (qwen3:8b) + ORCA tools -> Kokoro TTS -> Speaker
"""

import asyncio
import os
import sys

if sys.platform == "win32":
    _nvidia_root = os.path.join(sys.prefix, "Lib", "site-packages", "nvidia")
    if os.path.isdir(_nvidia_root):
        _dirs = []
        for _sub in os.listdir(_nvidia_root):
            _bin = os.path.join(_nvidia_root, _sub, "bin")
            if os.path.isdir(_bin):
                _dirs.append(_bin)
                try:
                    os.add_dll_directory(_bin)
                except (OSError, AttributeError):
                    pass
        if _dirs:
            os.environ["PATH"] = os.pathsep.join(_dirs + [os.environ.get("PATH", "")])
            print(f"[CUDA] Registered {len(_dirs)} NVIDIA DLL directories")

from loguru import logger

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import LLMRunFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.services.kokoro.tts import KokoroTTSService
from pipecat.services.ollama.llm import OLLamaLLMService
from pipecat.services.whisper.stt import WhisperSTTService, Model
from pipecat.transports.base_transport import TransportParams
from pipecat.transports.smallwebrtc.transport import SmallWebRTCTransport
from pipecat.workers.runner import WorkerRunner

from demo.voice.tools import FUNCTION_HANDLERS, ORCA_TOOLS_SCHEMA


SYSTEM_PROMPT = """You are ORCA, a marine intelligence voice assistant for fishermen and coastal operators along the Indian coast.

Rules:
- Keep answers SHORT — one or two spoken sentences unless the user asks for detail.
- Never use markdown, bullet lists, emojis, or code.
- Always call a tool to get real data. Never invent temperatures, wave heights, wind speeds, or coordinates.
- If the user names a place, call `orca_resolve_location` first to get lat/lon.
- For current conditions or "right now", call `orca_marine_snapshot`.
- For "tomorrow", "later", "forecast", or wind questions, call `orca_weather`.
- For safety or restriction questions, call `orca_geofence`.
- Distinguish clearly between OBSERVED and FORECAST data.
- If a tool fails or data is missing, say so plainly — never guess.
- End marine-conditions answers with a one-line practical recommendation when relevant.
"""

OLLAMA_MODEL = "qwen3:8b"
OLLAMA_URL = "http://127.0.0.1:11434/v1"

KOKORO_MODEL = os.path.expanduser("~/.cache/pipecat/kokoro-onnx/kokoro-v1.0.onnx")
KOKORO_VOICES = os.path.expanduser("~/.cache/pipecat/kokoro-onnx/voices-v1.0.bin")
KOKORO_VOICE = "af_heart"


async def run_bot(webrtc_connection):
    greeted = False

    transport = SmallWebRTCTransport(
        webrtc_connection=webrtc_connection,
        params=TransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            audio_in_sample_rate=16000,
            audio_out_sample_rate=24000,
        ),
    )

    logger.info("Loading Whisper...")
    try:
        stt = WhisperSTTService(
            device="cuda",
            compute_type="float16",
            ttfs_p99_latency=0.5,
            settings=WhisperSTTService.Settings(model=Model.DISTIL_MEDIUM_EN),
        )
    except Exception as exc:
        logger.warning(f"CUDA Whisper failed ({exc}); falling back to CPU")
        stt = WhisperSTTService(
            device="cpu",
            compute_type="int8",
            ttfs_p99_latency=1.0,
            settings=WhisperSTTService.Settings(model=Model.BASE_EN),
        )

    llm = OLLamaLLMService(
        settings=OLLamaLLMService.Settings(
            model=OLLAMA_MODEL,
            system_instruction=SYSTEM_PROMPT,
        ),
        base_url=OLLAMA_URL,
    )

    for name, handler in FUNCTION_HANDLERS.items():
        llm.register_function(name, handler)
        logger.info(f"[voice] registered tool: {name}")

    logger.info(f"Loading Kokoro from {KOKORO_MODEL}")
    tts = KokoroTTSService(
        model_path=KOKORO_MODEL,
        voices_path=KOKORO_VOICES,
        settings=KokoroTTSService.Settings(voice=KOKORO_VOICE),
    )

    vad = SileroVADAnalyzer(
        params=VADParams(
            confidence=0.5,
            start_secs=0.2,
            stop_secs=0.4,
            min_volume=0.2,
        )
    )

    context = LLMContext(tools=ORCA_TOOLS_SCHEMA)
    user_agg, assistant_agg = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(vad_analyzer=vad),
    )

    pipeline = Pipeline([
        transport.input(),
        stt,
        user_agg,
        llm,
        tts,
        transport.output(),
        assistant_agg,
    ])

    worker = PipelineWorker(
        pipeline,
        params=PipelineParams(
            audio_in_sample_rate=16000,
            audio_out_sample_rate=24000,
            enable_metrics=True,
            idle_timeout_secs=3600,
        ),
    )

    runner = WorkerRunner(handle_sigint=False)

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        nonlocal greeted
        if greeted:
            return
        greeted = True
        logger.info("=== CLIENT CONNECTED ===")
        await asyncio.sleep(1.0)
        context.add_message({
            "role": "developer",
            "content": (
                "Greet the user briefly as ORCA and ask which marine "
                "conditions they need. Do not call any tools for this greeting."
            ),
        })
        await worker.queue_frames([LLMRunFrame()])

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info("Client disconnected")
        await runner.cancel()

    await runner.add_workers(worker)
    await runner.run()