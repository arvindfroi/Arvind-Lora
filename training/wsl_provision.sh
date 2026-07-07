#!/usr/bin/env bash
# Step 2 of 2 — run INSIDE Ubuntu (WSL2) after the reboot:
#   bash /mnt/c/Users/arvin/Arvind-Lora/training/wsl_provision.sh
#
# Sets up a CUDA training env with the FAST hybrid-attention kernels that don't
# exist on native Windows, reuses the 19 GB model already on the D: drive
# (no re-download), verifies the fast path is really active, then runs the
# smoke-gated pipeline -> full 2-epoch Qwythos QLoRA.
set -euo pipefail

WIN_REPO="/mnt/c/Users/arvin/Arvind-Lora"
REPO="$HOME/Arvind-Lora"              # work on ext4 (fast), not /mnt/c
HF_CACHE="/mnt/d/LocalAI/hf-cache"    # reuse the already-downloaded weights
export HF_HOME="$HF_CACHE"

echo "== [1/6] GPU visible to WSL? =="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || {
  echo "!! nvidia-smi failed inside WSL. Update the Windows NVIDIA driver "
  echo "   (it provides the WSL CUDA driver) and re-run this script."; exit 1; }

echo "== [2/6] System packages =="
# Skip the sudo/apt step if the tools are already present (lets the script be
# re-run non-interactively without hanging on a sudo password prompt).
if command -v rsync >/dev/null && command -v gcc >/dev/null \
   && python3 -c "import venv" 2>/dev/null; then
    echo "   system packages already present — skipping apt."
else
    sudo apt-get update -y
    sudo apt-get install -y python3-venv python3-pip build-essential git rsync
fi

echo "== [3/6] Copy repo code+data to ext4 (small; keeps private data local) =="
mkdir -p "$REPO"
rsync -a --exclude '.venv' --exclude 'smoke_out' --exclude 'lora_out_qwythos' \
      --exclude 'pipeline.log' --exclude 'status_*.json' \
      "$WIN_REPO"/ "$REPO"/
cd "$REPO"

echo "== [4/6] Python env + CUDA wheels + FAST kernels =="
# /tmp is a small RAM-backed tmpfs; the multi-GB CUDA wheels overflow it.
# Point pip's temp + cache at the roomy ext4 root instead.
export TMPDIR="$REPO/.piptmp"; mkdir -p "$TMPDIR"
export PIP_CACHE_DIR="$REPO/.pipcache"; mkdir -p "$PIP_CACHE_DIR"
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip wheel
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install "transformers==5.13.0" datasets peft trl bitsandbytes accelerate
# The kernels that make the hybrid arch fast (Triton-based; no nvcc needed):
pip install flash-linear-attention || pip install fla-core || true
# Short-conv kernel — best effort (prebuilt wheel if available for this torch):
pip install causal-conv1d || echo "   (causal-conv1d wheel unavailable; fla still covers the main path)"

echo "== [5/6] Verify the fast path is actually active =="
python - <<'PY'
import torch, importlib.util
assert torch.cuda.is_available(), "CUDA not available in WSL"
print("torch", torch.__version__, "| GPU", torch.cuda.get_device_name(0))
has_fla = importlib.util.find_spec("fla") is not None
has_conv = importlib.util.find_spec("causal_conv1d") is not None
print("flash-linear-attention:", "OK" if has_fla else "MISSING")
print("causal-conv1d:", "OK" if has_conv else "missing (non-fatal)")
if not has_fla:
    raise SystemExit("!! flash-linear-attention missing — the fast path won't "
                     "engage. Fix the install before training.")
PY

echo "== [6/6] Smoke-gated training (measures step time, then full run) =="
# packing stays OFF: correct-by-construction. With fla the unpacked run is fast.
HF_HOME="$HF_CACHE" python training/run_pipeline.py --epochs 2

echo ""
echo "=================================================================="
echo " If it finished: adapter is at $REPO/training/lora_out_qwythos/adapter"
echo " Copy it back to Windows with:"
echo "   cp -r $REPO/training/lora_out_qwythos/adapter '$WIN_REPO/training/'"
echo "=================================================================="
