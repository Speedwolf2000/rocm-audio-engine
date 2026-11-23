# ROCm Audio & Forensic Provenance Engine

Open-source ROCm-native PyTorch container and benchmark suite for multi-model generative audio, HTDemucs-6s stem separation, and SHA-256/C2PA provenance on AMD Instinct GPUs (MI300X 192 GB HBM3).

## Benchmark Results (30s Stereo Generation + 6-Stem Separation)

| Metric | AMD MI300X (ROCm) | NVIDIA Baseline (CUDA) |
|---|---|---|
| Resident VRAM (Gen + HTDemucs-6s) | _pending_ | _pending_ |
| Generation Warm RTF | _pending_ | _pending_ |
| HTDemucs-6s Separation RTF | _pending_ | _pending_ |

## Run on AMD Instinct GPU Droplet (`gpu-amd-base`)

~~~~bash
docker run --rm --device=/dev/kfd --device=/dev/dri \
  --group-add video --ipc=host --shm-size 16G \
  --security-opt seccomp=unconfined \
  -v /scratch/hf_cache:/workspace/hf_cache \
  -v $(pwd)/results:/workspace/results \
  --env-file .env.local \
  -e HF_TOKEN=$HF_TOKEN \
  registry.digitalocean.com/radiotomy-registry/rocm-audio:v1
~~~~