"""REST and WebSocket service.

The problem statement asks for "REST/gRPC APIs and SDKs for integration with
core banking systems, contact centre platforms, enterprise communication tools,
and telecom networks".  Streamlit cannot serve that; this can.

Privacy properties, which are requirements rather than nice-to-haves:

* No audio is ever written to disk. Uploads are decoded in memory and the
  buffer is dropped when the request ends.
* Only scores, family probabilities and metadata are logged.
* Everything runs locally, so audio never leaves the operator's machine.

Run it with:  uvicorn src.api:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import io
import logging
import os
import time
import uuid
from typing import Any

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect, BackgroundTasks, Depends, Request
from fastapi.security import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src import audio_prep, families, model_loader
from src.inference import StreamingScorer, analyse
from src.risk import CallContext
from src.telephony_degrader import Condition, available_codecs, missing_codecs

logging.basicConfig(
    level=os.environ.get("VFD_LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
log = logging.getLogger("vfd.api")

MAX_UPLOAD_BYTES = int(os.environ.get("VFD_MAX_UPLOAD_BYTES", 25 * 1024 * 1024))
MAX_SECONDS = float(os.environ.get("VFD_MAX_SECONDS", 300))


from src.alerting import WorkflowEngine

API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)
EXPECTED_API_KEY = os.environ.get("VFD_API_KEY", "vfd-demo-key-2026")

async def verify_api_key(request: Request, api_key: str = Depends(api_key_header)):
    if api_key == EXPECTED_API_KEY:
        return api_key
    client_host = request.client.host if request.client else ""
    if client_host in ("127.0.0.1", "localhost", "::1"):
        return api_key
    raise HTTPException(status_code=403, detail="Could not validate credentials")

workflow_engine = WorkflowEngine()

app = FastAPI(
    title="Indic Voice Forensic Inspector",
    description="Real-time detection of AI-generated speech over Indian telephony channels.",
    version="1.0.0",
    dependencies=[Depends(verify_api_key)],
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("VFD_CORS_ORIGINS", "*").split(","),
    allow_methods=["*"],
    allow_headers=["*"],
)


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    checkpoint: str
    device: str
    codecs_available: dict[str, bool]
    codecs_missing: list[str]
    version: str


class EvidenceItem(BaseModel):
    name: str
    value: float | str
    unit: str
    reading: str


class AnalyzeResponse(BaseModel):
    request_id: str
    verdict: str = Field(description="Real human voice, Uncertain, or AI-generated voice")
    risk_score: float = Field(ge=0, le=100)
    band: str
    recommended_action: str
    confidence: float
    tier1_spoof_probability: float
    tier2_family: str
    tier2_family_label: str
    tier2_probabilities: dict[str, float]
    tier3_novelty_distance: float | None
    tier3_is_unknown_system: bool | None
    tier0_watermark: str
    speaker: dict | None = None
    risk_contributions: dict[str, float] = {}
    risk_reasons: list[str] = []
    evidence: list[EvidenceItem]
    window_scores: list[float]
    seconds_analysed: float
    latency_ms: float
    notes: list[str]


def _decode(raw: bytes) -> tuple[np.ndarray, int]:
    """Decode uploaded bytes to a mono float array, in memory only."""
    try:
        import soundfile as sf

        data, sr = sf.read(io.BytesIO(raw), dtype="float32")
    except Exception as exc:
        raise HTTPException(
            status_code=415,
            detail=f"Could not decode this audio. Send WAV or FLAC, mono or stereo. ({exc})",
        ) from exc
    x = audio_prep.to_mono(data)
    if len(x) / sr > MAX_SECONDS:
        raise HTTPException(status_code=413, detail=f"Audio longer than {MAX_SECONDS:g}s. Split it and retry.")
    return x, sr


@app.get("/health", response_model=HealthResponse, tags=["ops"])
def health() -> HealthResponse:
    """Liveness plus a truthful statement of what this instance can do."""
    import torch

    loaded = model_loader.checkpoint_available(model_loader.CHECKPOINT_DIR)
    return HealthResponse(
        status="ok" if loaded else "degraded",
        model_loaded=loaded,
        checkpoint=model_loader.CHECKPOINT_DIR,
        device="cuda" if torch.cuda.is_available() else "cpu",
        codecs_available=available_codecs(),
        codecs_missing=missing_codecs(),
        version=app.version,
    )


@app.get("/v1/families", tags=["reference"])
def list_families() -> dict[str, Any]:
    """The taxonomy the Tier 2 head predicts."""
    return {
        "count": families.NUM_FAMILIES,
        "note": "Class 0 is bonafide human speech; classes 1-6 are synthetic decoder families.",
        "families": [families.describe(f.idx) for f in families.FAMILIES],
    }


@app.get("/v1/codecs", tags=["reference"])
def list_codecs() -> dict[str, Any]:
    """Telephony conditions this instance can simulate."""
    return {
        "available": available_codecs(),
        "missing": missing_codecs(),
        "hint": "Missing codecs need an ffmpeg build with libopencore-amrnb, libgsm and libopus.",
    }


@app.post("/v1/analyze", response_model=AnalyzeResponse, tags=["detection"])
async def analyze(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="WAV or FLAC audio"),
    simulate_codec: str | None = Form(None, description="Optional telephony codec to apply first"),
    simulate_bitrate_kbps: float | None = Form(None),
    simulate_packet_loss: float = Form(0.0, ge=0.0, le=1.0),
    speaker_id: str | None = Form(None, description="Identity the caller claims, for the cross-session check"),
    number_is_known: bool | None = Form(None),
    first_contact: bool = Form(False),
    transaction_value_inr: float | None = Form(None),
    is_privileged_action: bool = Form(False),
    outside_business_hours: bool = Form(False),
    prior_fraud_flag: bool = Form(False),
    webhook_url: str | None = Form(None),
    smtp_email: str | None = Form(None),
) -> AnalyzeResponse:
    """Analyse one clip.

    The audio is decoded in memory and discarded when this call returns. It is
    never written to disk.
    """
    rid = uuid.uuid4().hex[:12]
    started = time.perf_counter()

    raw = await file.read()
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail=f"Upload exceeds {MAX_UPLOAD_BYTES} bytes.")
    if not raw:
        raise HTTPException(status_code=400, detail="Empty upload.")

    x, sr = _decode(raw)
    del raw

    cond = None
    if simulate_codec:
        cond = Condition(
            codec=simulate_codec,
            bitrate_kbps=simulate_bitrate_kbps,
            packet_loss=simulate_packet_loss,
        )

    ctx = CallContext(
        number_is_known=number_is_known,
        first_contact=first_contact,
        claimed_identity=speaker_id,
        transaction_value_inr=transaction_value_inr,
        is_privileged_action=is_privileged_action,
        outside_business_hours=outside_business_hours,
        prior_fraud_flag=prior_fraud_flag,
    )

    try:
        result = analyse(x, sr, apply_degradation=cond, speaker_id=speaker_id, context=ctx)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    background_tasks.add_task(
        workflow_engine.evaluate,
        request_id=rid,
        risk_score=result.risk_score,
        is_fake=result.verdict == "AI-generated voice",
        context={"webhook_url": webhook_url, "smtp_email": smtp_email}
    )

    log.info(
        "rid=%s verdict=%s risk=%.1f family=%s secs=%.1f latency=%.0fms",
        rid, result.verdict, result.risk_score, result.tier2_family,
        result.seconds_analysed, (time.perf_counter() - started) * 1000,
    )
    return AnalyzeResponse(request_id=rid, **result.to_dict())


@app.get("/v1/speakers", tags=["speaker"])
def list_speakers() -> dict[str, Any]:
    """Everyone enrolled for cross-session verification.

    The store holds 192-dimension voiceprints and metadata only. It contains no
    audio, and an embedding cannot be inverted back to a waveform.
    """
    from src.speaker import EnrollmentStore

    speakers = EnrollmentStore().list_speakers()
    return {
        "count": len(speakers),
        "speakers": speakers,
        "note": "Voiceprints only. No audio is stored at any point.",
    }


@app.post("/v1/speakers/{speaker_id}/enroll", tags=["speaker"])
async def enroll_speaker(
    speaker_id: str,
    files: list[UploadFile] = File(..., description="Three or more genuine clips, from separate occasions"),
    display_name: str = Form(""),
    channel: str = Form("unknown"),
) -> dict[str, Any]:
    """Enrol a known person from historical genuine samples.

    This is **not** training. Each clip is one forward pass through a
    pretrained VoxCeleb model; the audio is discarded as soon as the voiceprint
    is computed and is never written to disk.

    Use clips from separate occasions rather than slices of one recording - the
    store needs to learn how this person naturally varies, and three slices of
    one call teach it nothing.
    """
    from src.speaker import MIN_ENROLL_CLIPS, enroll_from_audio

    if len(files) < MIN_ENROLL_CLIPS:
        raise HTTPException(
            status_code=400,
            detail=f"Need at least {MIN_ENROLL_CLIPS} clips from separate occasions, got {len(files)}.",
        )
    clips = []
    for f in files:
        raw = await f.read()
        if len(raw) > MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail=f"{f.filename} exceeds the size limit.")
        clips.append(_decode(raw))
        del raw

    try:
        rec = enroll_from_audio(speaker_id, clips, display_name or speaker_id, channel=channel)
    except ImportError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    finally:
        clips.clear()

    mean, std = rec.self_similarity()
    log.info("enrolled speaker_id=%s sessions=%d", speaker_id, len(rec.sessions))
    return {
        "speaker_id": rec.speaker_id,
        "display_name": rec.display_name,
        "sessions": len(rec.sessions),
        "ready": len(rec.sessions) >= MIN_ENROLL_CLIPS,
        "self_similarity_mean": round(mean, 4),
        "self_similarity_std": round(std, 4),
        "note": "Voiceprints stored. Audio was not retained.",
    }


@app.delete("/v1/speakers/{speaker_id}", tags=["speaker"])
def delete_speaker(speaker_id: str) -> dict[str, Any]:
    """Erase an enrolled voiceprint. Right-to-erasure in one call."""
    from src.speaker import EnrollmentStore

    return {"speaker_id": speaker_id, "deleted": EnrollmentStore().delete(speaker_id)}


@app.websocket("/v1/stream")
async def stream(ws: WebSocket) -> None:
    """Live call analysis.

    Send binary frames of 16 kHz mono float32 PCM. A JSON score is returned
    roughly once per second, after at least 1.5 seconds of speech has arrived.
    Send the text ``reset`` to start a new call on the same socket.
    """
    await ws.accept()
    rid = uuid.uuid4().hex[:12]
    try:
        scorer = StreamingScorer()
    except FileNotFoundError as exc:
        await ws.send_json({"error": str(exc), "code": "no_checkpoint"})
        await ws.close(code=1011)
        return

    log.info("rid=%s stream opened", rid)
    try:
        while True:
            msg = await ws.receive()
            if msg.get("type") == "websocket.disconnect":
                break
            if (text := msg.get("text")) is not None:
                if text.strip().lower() == "reset":
                    scorer.reset()
                    await ws.send_json({"status": "reset"})
                continue
            data = msg.get("bytes")
            if not data:
                continue
            chunk = np.frombuffer(data, dtype=np.float32)
            out = scorer.push(chunk, audio_prep.TARGET_SR)
            if out is not None:
                out["request_id"] = rid
                await ws.send_json(out)
    except WebSocketDisconnect:
        pass
    except Exception as exc:  # keep the socket's failure visible but contained
        log.exception("rid=%s stream error", rid)
        try:
            await ws.send_json({"error": str(exc), "code": "stream_error"})
        except RuntimeError:
            pass
    finally:
        log.info("rid=%s stream closed after %d windows", rid, len(scorer.history))


@app.post("/v1/detect", tags=["detection"])
async def detect_nextjs(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="WAV or FLAC audio"),
    webhook_url: str | None = Form(None),
    smtp_email: str | None = Form(None),
    channel: str | None = Form(None)
):
    try:
        if not file.filename.lower().endswith((".wav", ".mp3", ".ogg")):
            raise HTTPException(status_code=400, detail="Only .wav, .mp3, or .ogg files are supported.")
            
        raw = await file.read()
        if len(raw) > MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail=f"Upload exceeds {MAX_UPLOAD_BYTES} bytes.")
        if not raw:
            raise HTTPException(status_code=400, detail="Empty upload.")

        x, sr = _decode(raw)
        del raw

        try:
            cond = None
            if channel and channel != "none":
                cond = Condition(codec=channel.replace("-", "_"))
            result = analyse(x, sr, apply_degradation=cond)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
            
        data = result.to_dict()
        rid = uuid.uuid4().hex[:12]
        data["request_id"] = rid
        
        background_tasks.add_task(
            workflow_engine.evaluate, 
            request_id=rid, 
            risk_score=data["risk_score"], 
            is_fake=data["tier1_spoof_probability"] > 0.5,
            context={"webhook_url": webhook_url, "smtp_email": smtp_email}
        )
        
        return data
    except HTTPException:
        raise
    except Exception:
        # Log the traceback, return an opaque id.  Absolute filesystem paths and
        # internal structure do not belong in an HTTP response body.
        err_id = uuid.uuid4().hex[:12]
        log.exception("unhandled error in /v1/detect [%s]", err_id)
        raise HTTPException(
            status_code=500,
            detail=f"Analysis failed. Quote reference {err_id} when reporting this.",
        )
