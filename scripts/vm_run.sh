#!/bin/bash
# Runs on a GPU VM: usage vm_run.sh <base|edit> <gs://bucket/prefix>
# Installs vLLM, fetches the model, runs nothink, think, force; finished work is skipped on restart.
# out/ is copied to the bucket every 2 minutes, so a preempted VM loses at most one 100-problem chunk.
m=$1; dest=$2; set -x; cd ~
export PATH=~/venv/bin:/usr/local/cuda/bin:$PATH CUDA_HOME=/usr/local/cuda   # vLLM JIT-builds kernels with ninja + nvcc
[ -x venv/bin/python ] || { curl -LsSf https://astral.sh/uv/install.sh | sh; ~/.local/bin/uv venv venv -p 3.12 && ~/.local/bin/uv pip install -p venv/bin/python vllm math-verify datasets ninja; }
repo=$([ $m = base ] && echo Qwen/Qwen3.5-0.8B || echo RikiyaT/Qwen3.5-0.8B-weight-space-programming)
[ -d models/$m ] || venv/bin/hf download $repo --local-dir models/$m
mkdir -p out/$m; gcloud storage rsync $dest/$m out/$m 2>/dev/null   # resume after a preemption
( while sleep 120; do gcloud storage rsync out/$m $dest/$m -q; done ) &
for mode in nothink think force; do
  [ -f out/$m/$mode.summary.json ] || venv/bin/python math500_vllm.py models/$m out/$m $mode
done
gcloud storage rsync out/$m $dest/$m; echo ALL_DONE
