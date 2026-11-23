import hashlib
import json
import os
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import boto3
import soundfile as sf
import torch
from demucs.apply import apply_model
from demucs.pretrained import get_model
from diffusers import StableAudioPipeline
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

OUT_DIR = Path(os.getenv("OUT_DIR", "/workspace/results"))
OUT_DIR.mkdir(parents=True, exist_ok=True)
STEM_NAMES_6S = ["drums", "bass", "other", "vocals", "guitar", "piano"]

app = FastAPI(title="ROCm Audio & Provenance Worker", version="0.2.0")

_state = {
    "device": None,
    "pipe": None,
    "demucs": None,
    "loaded": False,
    "audio_model_id": os.getenv("AUDIO_MODEL_ID", "stabilityai/stable-audio-open-1.0"),
    "demucs_model_id": "htdemucs_6s",
}


class AudioJobRequest(BaseModel):
    prompt: str = Field(..., min_length=3, max_length=500)
    negative_prompt: str = Field(default="low quality, distorted, mono", max_length=300)
    duration_sec: float = Field(default=30.0, ge=5.0, le=47.0)
    steps: int = Field(default=100, ge=10, le=200)
    separate_stems: bool = Field(default=True)
    upload_spaces: bool = Field(default=True)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def upload_to_spaces(files: list[Path], job_id: str) -> list[str]:
    bucket = os.getenv("SPACES_BUCKET")
    key = os.getenv("SPACES_KEY")
    secret = os.getenv("SPACES_SECRET")
    endpoint = os.getenv("SPACES_ENDPOINT", "https://atl1.digitaloceanspaces.com")
    region = os.getenv("SPACES_REGION", "atl1")
    if not all([bucket, key, secret]):
        return []

    s3 = boto3.client(
        "s3",
        region_name=region,
        endpoint_url=endpoint,
        aws_access_key_id=key,
        aws_secret_access_key=secret,
    )
    uploaded = []
    for path in files:
        object_key = f"jobs/{job_id}/{path.name}"
        s3.upload_file(str(path), bucket, object_key)
        uploaded.append(object_key)
    return uploaded


@app.on_event("startup")
def load_models():
    if not torch.cuda.is_available():
        return
    device = torch.device("cuda:0")
    pipe = StableAudioPipeline.from_pretrained(
        _state["audio_model_id"],
        torch_dtype=torch.float16,
    ).to(device)
    demucs_model = get_model(_state["demucs_model_id"]).to(device).eval()
    _state.update({"device": device, "pipe": pipe, "demucs": demucs_model, "loaded": True})


@app.get("/health")
def health():
    gpu_available = torch.cuda.is_available()
    return {
        "status": "ok",
        "gpu_available": gpu_available,
        "models_loaded": _state["loaded"],
        "device_name": torch.cuda.get_device_name(0) if gpu_available else None,
        "hip_version": torch.version.hip,
    }


@app.post("/jobs/generate")
def generate_audio(req: AudioJobRequest):
    if not _state["loaded"] or _state["pipe"] is None:
        raise HTTPException(status_code=503, detail="GPU audio pipeline not loaded on this host.")

    job_id = str(uuid.uuid4())[:8]
    created_at = datetime.now(timezone.utc).isoformat()
    device = _state["device"]
    pipe = _state["pipe"]

    t0 = time.perf_counter()
    with torch.inference_mode():
        audio_tensor = pipe(
            req.prompt,
            negative_prompt=req.negative_prompt,
            num_inference_steps=req.steps,
            audio_end_in_s=req.duration_sec,
            num_waveforms_per_prompt=1,
        ).audios[0]
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    gen_sec = round(time.perf_counter() - t0, 3)

    wav_path = OUT_DIR / f"{job_id}_master.wav"
    sr = pipe.vae.sampling_rate
    sf.write(wav_path, audio_tensor.T.float().cpu().numpy(), sr)
    master_sha256 = sha256_file(wav_path)

    artifacts = [wav_path]
    stem_records = []
    sep_sec: Optional[float] = None
    stem_count = 0

    if req.separate_stems and _state["demucs"] is not None:
        mix = audio_tensor.float().unsqueeze(0).to(device)
        if mix.shape[1] == 1:
            mix = mix.repeat(1, 2, 1)
        t_sep = time.perf_counter()
        with torch.inference_mode():
            stems = apply_model(_state["demucs"], mix, device=device, split=True)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        sep_sec = round(time.perf_counter() - t_sep, 3)
        stem_batch = stems[0].detach().cpu()
        stem_count = int(stem_batch.shape[0])

        for idx in range(stem_count):
            stem_label = STEM_NAMES_6S[idx] if idx < len(STEM_NAMES_6S) else f"stem_{idx}"
            stem_path = OUT_DIR / f"{job_id}_stem_{idx:02d}_{stem_label}.wav"
            sf.write(stem_path, stem_batch[idx].T.float().numpy(), sr)
            stem_hash = sha256_file(stem_path)
            artifacts.append(stem_path)
            stem_records.append(
                {
                    "index": idx,
                    "stem": stem_label,
                    "file_name": stem_path.name,
                    "sha256": stem_hash,
                }
            )

    manifest_path = OUT_DIR / f"{job_id}_provenance_manifest.json"
    manifest = {
        "schema_version": "rocm-provenance-manifest-v1",
        "job_id": job_id,
        "created_at_utc": created_at,
        "models": {
            "generator": _state["audio_model_id"],
            "stem_separator": _state["demucs_model_id"] if req.separate_stems else None,
            "sample_rate_hz": sr,
        },
        "generation_params": {
            "prompt": req.prompt,
            "negative_prompt": req.negative_prompt,
            "duration_sec": req.duration_sec,
            "num_inference_steps": req.steps,
        },
        "timings_sec": {
            "generation": gen_sec,
            "stem_separation": sep_sec,
        },
        "master_artifact": {
            "file_name": wav_path.name,
            "sha256": master_sha256,
        },
        "stem_artifacts": stem_records,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))
    artifacts.append(manifest_path)

    uploaded_keys = upload_to_spaces(artifacts, job_id) if req.upload_spaces else []
    if uploaded_keys:
        manifest["spaces_keys"] = uploaded_keys
        manifest_path.write_text(json.dumps(manifest, indent=2))

    return {
        "job_id": job_id,
        "generation_sec": gen_sec,
        "separation_sec": sep_sec,
        "stem_count": stem_count,
        "master_wav_path": str(wav_path),
        "master_sha256": master_sha256,
        "manifest_path": str(manifest_path),
        "stems": stem_records,
        "spaces_bucket": os.getenv("SPACES_BUCKET"),
        "spaces_keys": uploaded_keys,
    }