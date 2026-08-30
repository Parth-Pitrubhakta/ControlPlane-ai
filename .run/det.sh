#!/usr/bin/env bash
# Detector service: one coordinator on 8100 plus one worker process per model.
#
# The split is not premature optimisation. These models are kernel-launch bound,
# so a forward pass holds the GIL for its whole duration and three detectors in
# one process serialise no matter which GPU they sit on (measured: separate GPUs
# changed nothing, 144 ms vs 143 ms). Separate processes are what make the
# tier-1 budget reachable. Set DET_WORKERS empty to run everything in-process.
#
# vLLM owns physical GPU 3. GPU 0 carries another user's job, so det takes 1 and 2.
set -euo pipefail
cd "$(dirname "$0")/.."
R="$PWD"
E=/DATA2/home/parth/.conda/envs/cp-vllm
export HF_HOME=${HF_HOME:-/DATA2/shared/hf-cache}
export LD_LIBRARY_PATH=$E/lib:${LD_LIBRARY_PATH:-}
export PYTHONPATH="$R"
export LOG_LEVEL=${LOG_LEVEL:-INFO}
# Preselection stays OFF. It was tried on ControlPlane-Bench as a fix for
# contradiction false positives and made things worse, not better: alerts fell
# 748 to 406 per 1000 only because recall collapsed (contradicted 0.297 -> 0.027,
# catch rate 0.937 -> 0.60) and p50 rose 192 -> 290 ms. Embedding similarity
# picks chunks that resemble the sentence, which is not the same as chunks that
# could refute it. The threshold, not the candidate set, is the right lever.
export NLI_TOPK=${NLI_TOPK:-0}
mkdir -p "$R/.run/log"

start_worker() {   # role port cuda_device
  CUDA_VISIBLE_DEVICES=$3 DET_ROLE=$1 DET_DEVICE=cuda \
    nohup $E/bin/python -m uvicorn det.worker:app --host 127.0.0.1 --port $2 \
    > "$R/.run/log/det-$1.log" 2>&1 &
  echo $! > "$R/.run/det-$1.pid"
  echo "  worker $1 -> port $2, gpu $3"
}

start_worker nli    8101 1
start_worker safety 8102 2
start_worker bias   8103 2

for p in 8101 8102 8103; do
  for i in $(seq 1 90); do
    curl -sf "http://127.0.0.1:$p/health" >/dev/null 2>&1 && break
    sleep 1
  done
done

export DET_WORKERS=${DET_WORKERS:-nli=8101,safety=8102,bias=8103}
# tier 2 runs in the coordinator but reuses the NLI model already loaded in its
# worker, rather than loading a second copy
export T2_NLI_URL=${T2_NLI_URL:-http://127.0.0.1:8101}
export VLLM_URL=${VLLM_URL:-http://127.0.0.1:8000/v1}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-2}
export DET_DEVICE=${DET_DEVICE:-cuda}
echo $$ > "$R/.run/det-serve.pid"
exec $E/bin/python -m uvicorn det.serve:app --host 127.0.0.1 --port 8100
