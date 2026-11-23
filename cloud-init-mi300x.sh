#!/bin/bash
set -euo pipefail
exec > >(tee -a /var/log/rocm-bootstrap.log) 2>&1

apt-get update && apt-get install -y --no-install-recommends git ca-certificates curl
mkdir -p /scratch/hf_cache /workspace/results
chmod 777 /scratch/hf_cache /workspace/results

cd /opt
if [ ! -d /opt/rocm-audio-engine ]; then
  git clone https://github.com/Speedwolf2000/rocm-audio-engine.git
fi
cd /opt/rocm-audio-engine

# Optional: populate /opt/rocm-audio-engine/.env.local and HF_TOKEN before running docker
echo "Bootstrap complete. Ready to authenticate DOCR, set HF_TOKEN, build, and run benchmark."