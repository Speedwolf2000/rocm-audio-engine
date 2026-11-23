import hashlib
import json
import os
import time
from pathlib import Path

import boto3
import soundfile as sf
import torch
from demucs.apply import apply_model
from demucs.pretrained import get_model
from diffusers import StableAudioPipeline

OUT_DIR = Path("/workspace/results")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def vram_gb():
    return round(torch.cuda.memory_allocated() / (1024**3), 3)


def peak_vram_gb():
    return round(torch.cuda.max_memory_allocated() / (1024**3), 3)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def upload_to_spaces(files: list[Path]) -> list[str]:
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
    run_id = time.strftime("%Y%m%d-%H%M%S")
    uploaded = []
    for path in files:
        s3_key = f"benchmarks/{run_id}/{path.name}"
        s3.upload_file(str(path), bucket, s3_key)
        uploaded.append(s3_key)
    return uploaded


def main():
    assert torch.cuda.is_available(), "No ROCm/CUDA GPU detected."
    device = torch.device("cuda:0")
    props = torch.cuda.get_device_properties(0)

    report = {
        "hardware": {
            "device_name": props.name,
            "total_vram_gb": round(props.total_memory / (1024**3), 2),
            "torch_version": torch.__version__,
            "hip_version": torch.version.hip,
            "cuda_version": torch.version.cuda,
        },
        "benchmarks": {},
    }

    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    pipe = StableAudioPipeline.from_pretrained(
        os.getenv("AUDIO_MODEL_ID", "stabilityai/stable-audio-open-1.0"),
        torch_dtype=torch.float16,
    ).to(device)
    sync()
    gen_load_sec = round(time.perf_counter() - t0, 3)

    t0 = time.perf_counter()
    demucs_model = get_model("htdemucs_6s").to(device).eval()
    sync()
    demucs_load_sec = round(time.perf_counter() - t0, 3)
    resident_vram = vram_gb()

    audio_len_sec = 30.0
    prompt = "Modern Red Dirt country rock groove, acoustic guitar, warm bass, punchy drums, 112 BPM"

    def run_gen():
        torch.cuda.reset_peak_memory_stats()
        t_start = time.perf_counter()
        with torch.inference_mode():
            out = pipe(
                prompt,
                negative_prompt="low quality, distorted, mono",
                num_inference_steps=100,
                audio_end_in_s=audio_len_sec,
                num_waveforms_per_prompt=1,
            ).audios[0]
        sync()
        return out, time.perf_counter() - t_start, peak_vram_gb()

    _, cold_gen_sec, _ = run_gen()
    audio_tensor, warm_gen_sec, gen_peak_vram = run_gen()

    wav_path = OUT_DIR / "rocm_generated_30s.wav"
    sr = pipe.vae.sampling_rate
    sf.write(wav_path, audio_tensor.T.float().cpu().numpy(), sr)
    master_sha256 = sha256_file(wav_path)

    mix = audio_tensor.float().unsqueeze(0).to(device)
    if mix.shape[1] == 1:
        mix = mix.repeat(1, 2, 1)

    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    with torch.inference_mode():
        stems = apply_model(demucs_model, mix, device=device, split=True)
    sync()
    sep_sec = time.perf_counter() - t0
    sep_peak_vram = peak_vram_gb()

    report["benchmarks"] = {
        "models_resident_vram_gb": resident_vram,
        "generation": {
            "cold_load_sec": gen_load_sec,
            "first_run_sec": round(cold_gen_sec, 3),
            "warm_run_sec": round(warm_gen_sec, 3),
            "audio_duration_sec": audio_len_sec,
            "rtf_warm": round(warm_gen_sec / audio_len_sec, 4),
            "peak_vram_gb": gen_peak_vram,
            "output_sha256": master_sha256,
        },
        "stem_separation_htdemucs_6s": {
            "cold_load_sec": demucs_load_sec,
            "separation_sec": round(sep_sec, 3),
            "rtf": round(sep_sec / audio_len_sec, 4),
            "stems_count": int(stems.shape[1]),
            "peak_vram_gb": sep_peak_vram,
        },
    }

    report_path = OUT_DIR / "rocm_benchmark.json"
    report_path.write_text(json.dumps(report, indent=2))
    uploaded = upload_to_spaces([wav_path, report_path])
    if uploaded:
        report["spaces_uploads"] = uploaded
        report_path.write_text(json.dumps(report, indent=2))

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()