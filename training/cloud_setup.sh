#!/usr/bin/env bash
# Turnkey Qwythos LoRA training on a rented cloud GPU (RunPod / Vast / Lambda).
# Paste this whole thing into a fresh pod's terminal, or:
#   export GITHUB_TOKEN=ghp_xxx        # fine-grained PAT with read access to the private repo
#   bash cloud_setup.sh
#
# Recommended pod: 1x RTX 4090 24GB (or A40/L40 48GB), "PyTorch 2.x" template,
# ~50GB disk. Datacenter GPUs have tuned fla + flash-attn kernels, so this runs
# ~1-2h for 2 epochs (vs ~40h on the Blackwell laptop). DELETE THE VOLUME after
# you download the adapter — the corpus is private.
set -euo pipefail

REPO_URL="github.com/arvindfroi/Arvind-Lora"
: "${GITHUB_TOKEN:?Set GITHUB_TOKEN to a PAT with read access to the private repo}"

echo "== [1/5] GPU =="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

echo "== [2/5] Clone private repo =="
cd /workspace 2>/dev/null || cd ~
if [ ! -d Arvind-Lora ]; then
    git clone "https://${GITHUB_TOKEN}@${REPO_URL}" Arvind-Lora
fi
cd Arvind-Lora

echo "== [3/5] Python env + kernels (fla + flash-attn) =="
pip install -q --upgrade pip
# Most PyTorch pods already ship a CUDA torch; only install if missing.
python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" \
  || pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -q "transformers==5.13.0" datasets peft trl bitsandbytes accelerate
pip install -q flash-linear-attention
# flash-attn: prebuilt wheels exist for datacenter GPUs (Ampere/Ada) — enables
# correct + fast packing on this hybrid arch.
pip install -q flash-attn --no-build-isolation || echo "flash-attn wheel unavailable; will train without packing"

echo "== [4/5] Build training data (balanced mix) =="
python training/prepare_training_data.py --max-chat-per-file 3500

echo "== [5/5] Train (packing + flash-attn if available) =="
HAS_FA=$(python -c "import importlib.util as u;print(1 if u.find_spec('flash_attn') else 0)")
if [ "$HAS_FA" = "1" ]; then
    python -u training/train_qwythos.py --epochs 2 \
        --packing --attn flash_attention_2 \
        --out training/lora_out_qwythos --status-file training/status.json
else
    python -u training/train_qwythos.py --epochs 2 \
        --out training/lora_out_qwythos --status-file training/status.json
fi

echo ""
echo "=================================================================="
echo " DONE. Adapter: training/lora_out_qwythos/adapter"
echo " Download it, THEN delete this pod's volume. Options:"
echo "   A) zip + download via the pod file browser:"
echo "        cd training/lora_out_qwythos && zip -r adapter.zip adapter"
echo "   B) push to your HF account (private):"
echo "        huggingface-cli login && \\"
echo "        huggingface-cli upload arvindfroi/arvind-qwythos-lora adapter"
echo "=================================================================="
