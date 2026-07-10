#!/usr/bin/env bash
# Turnkey Qwythos LoRA training on a rented cloud GPU.
#
#   export GITHUB_TOKEN=github_pat_xxx   # fine-grained PAT, read access to the private repo
#   bash Arvind-Lora/training/cloud_setup.sh
#
# Recommended pod: RunPod SECURE Cloud, 1x A40 48GB, EU region, "PyTorch 2.x"
# template, ~60GB disk. Secure Cloud = RunPod's own datacenters (not peer hosts);
# an EU region keeps this Norwegian personal data in the EU. Datacenter GPUs have
# tuned fla + flash-attn kernels, so this runs ~2-3h for 2 epochs (vs ~40h on the
# Blackwell laptop). DELETE THE VOLUME after you download the adapter — the corpus
# is verbatim private email and chat.
#
# Safe to re-run: the repo, the pip installs and the 19GB model download are all
# resumable, so if the pod restarts just run this again.
set -euo pipefail

REPO_URL="github.com/arvindfroi/Arvind-Lora"
WORK=/workspace
[ -d "$WORK" ] || WORK="$HOME"
: "${GITHUB_TOKEN:?Set GITHUB_TOKEN to a PAT with read access to the private repo}"

# Keep the 19GB model on the persistent volume, not the container's ephemeral /root.
# A pod restart otherwise re-downloads it, which costs real money.
export HF_HOME="$WORK/hf-cache"
mkdir -p "$HF_HOME"

# A closed browser tab sends SIGHUP and kills a 2-hour run. Re-exec inside tmux so
# training survives; reattach later with `tmux attach -t train`.
if [ -z "${TMUX:-}" ] && [ "${NO_TMUX:-0}" != "1" ] && command -v tmux >/dev/null; then
    echo "== Re-executing inside tmux (session: train) so a dropped connection can't kill the run."
    echo "== Detach with Ctrl-b d, reattach with: tmux attach -t train"
    sleep 2
    exec tmux new-session -s train "GITHUB_TOKEN='$GITHUB_TOKEN' NO_TMUX=1 bash '$0'; echo; echo '[exited — press enter]'; read"
fi

echo "== [1/6] GPU =="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
VRAM_GB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
if [ "$VRAM_GB" -lt 22000 ]; then
    echo "!! ${VRAM_GB}MiB VRAM is too small for a 9B QLoRA. Use a >=24GB card." >&2
    exit 1
fi

echo "== [2/6] Clone private repo =="
cd "$WORK"
if [ ! -d Arvind-Lora ]; then
    git clone --depth 1 "https://${GITHUB_TOKEN}@${REPO_URL}" Arvind-Lora
fi
cd Arvind-Lora

echo "== [3/6] Python env + kernels (fla + flash-attn) =="
pip install -q --upgrade pip
python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" \
  || pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -q "transformers==5.13.0" datasets peft trl bitsandbytes accelerate
pip install -q ninja packaging          # flash-attn's build needs both
pip install -q flash-linear-attention
# flash-attn's nvcc build OOMs the box if it fans out to every core.
MAX_JOBS=4 pip install -q flash-attn --no-build-isolation \
  || echo "!! flash-attn unavailable — will train WITHOUT packing (slower, still correct)"

# Hard-fail now rather than after the 19GB download: without fla's Triton kernels
# 24 of 32 layers fall back to a torch loop and a 2h run becomes a 40h run.
# (causal-conv1d is deliberately NOT required — it only gates single-token decoding,
# not training. It was a red herring that cost hours locally.)
python - <<'PY'
import importlib, sys
for mod in ("fla.ops.gated_delta_rule", "fla.ops.delta_rule", "fla.ops"):
    try:
        m = importlib.import_module(mod)
    except Exception:
        continue
    if hasattr(m, "chunk_gated_delta_rule"):
        print(f"   fla fast kernels: OK ({mod}.chunk_gated_delta_rule)")
        sys.exit(0)
sys.exit("FATAL: flash-linear-attention has no chunk_gated_delta_rule. "
         "Aborting before the 19GB model download.")
PY

echo "== [4/6] Build training data (balanced mix) =="
# chatgpt is capped tighter than the rest: 5082 raw samples of Arvind prompting an
# AI would otherwise crowd out the Snapchat dialect core.
python training/prepare_training_data.py --max-chat-per-file 3500 --cap chatgpt=2500
test -s training/out/train.jsonl || { echo "!! train.jsonl is empty" >&2; exit 1; }

echo "== [5/6] Train: smoke -> gate -> full run =="
HAS_FA=$(python -c "import importlib.util as u;print(1 if u.find_spec('flash_attn') else 0)")
# >=40GB cards fit a bigger batch without gradient checkpointing.
if [ "$VRAM_GB" -ge 40000 ]; then BATCH=4; GA=4; else BATCH=1; GA=16; fi
echo "   ${VRAM_GB}MiB VRAM -> batch=${BATCH} grad_accum=${GA}, flash_attn=${HAS_FA}"

ARGS=(--epochs 2 --batch "$BATCH" --grad-accum "$GA"
      --out training/lora_out_qwythos)
# Packing REQUIRES FlashAttention on this hybrid arch — without varlen cu_seqlens
# the packed samples bleed into each other, which is fatal for style training.
[ "$HAS_FA" = "1" ] && ARGS+=(--packing --attn flash_attention_2)

# The smoke stage runs the same batch/kernel path, then gates on finite loss and
# step time before committing to the paid 2-epoch run.
python -u training/run_pipeline.py "${ARGS[@]}" 2>&1 | tee "$WORK/train.log"

echo "== [6/6] Package the adapter =="
cd training/lora_out_qwythos
zip -qr "$WORK/adapter.zip" adapter
cd "$WORK"
du -h adapter.zip

cat <<EOF

==================================================================
 DONE. Adapter: $WORK/adapter.zip   (log: $WORK/train.log)

 1. Download adapter.zip via the pod's file browser.
 2. THEN terminate the pod AND delete the volume.
    The corpus is private personal data; it must not outlive this run.
==================================================================
EOF
