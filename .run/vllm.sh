#!/usr/bin/env bash
# Application model under test, and the T3 judge. Pinned to GPU 3.
# gpu-memory-utilization is 0.55, not the spec's 0.35: this card is shared and
# other processes already hold ~25% of it, so 0.35 leaves too little for weights.
set -euo pipefail
export CUDA_VISIBLE_DEVICES=3
# the env ships a newer libstdc++ than the system one; without this, sqlite3
# fails to load libicui18n (CXXABI_1.3.15 not found) and vllm serve dies on import
export LD_LIBRARY_PATH=/DATA2/home/parth/.conda/envs/cp-vllm/lib:${LD_LIBRARY_PATH:-}
# system nvcc is CUDA 11.5 and cannot target sm_90a, so FlashInfer's JIT build
# of the sampling kernels fails. Fall back to the torch sampler.
export VLLM_USE_FLASHINFER_SAMPLER=0

exec /DATA2/home/parth/.conda/envs/cp-vllm/bin/vllm serve Qwen/Qwen2.5-7B-Instruct \
  --port 8000 --host 127.0.0.1 \
  --gpu-memory-utilization 0.55 --max-model-len 8192 \
  --served-model-name Qwen/Qwen2.5-7B-Instruct
