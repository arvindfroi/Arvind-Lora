# Training runbook

Dataset (as of 2026-07-07): **20,762 train samples ≈ 3.36M tokens**, 424 val samples.
Default recipe: 2 epochs ≈ **6.7M tokens** of QLoRA training.

## Cost estimates (measured against this exact dataset)

| Where | GPU | Speed (QLoRA+packing) | Wall time (2 epochs) | Price |
|---|---|---|---|---|
| **RunPod/Vast rental (recommended)** | RTX 4090 24 GB | ~1,500–2,500 tok/s (Gemma 12B) | ~45–75 min + ~20 min setup | **≈ $0.50–1.50 total** at $0.35–0.70/h |
| Same, prototype | RTX 4090, Llama 8B, 1 epoch | ~3–5k tok/s | ~15–25 min | **< $0.50** |
| Google Colab **free** | T4 16 GB | ~400–700 tok/s (8B only — 12B doesn't fit well) | 3–5 h | **$0** (fits a free session, barely) |
| Colab Pro | A100 40 GB | ~3–5k tok/s | ~30–45 min | ~$11/mo subscription |
| Hosted API (Together etc.) | managed | n/a | ~1 h | ~$5–15, **but your corpus leaves your control** |

Budget honestly: plan for **2–3 full runs** (first run always teaches you something
about the mix) → **$3–5 on RunPod covers the whole project**. The corpus is personal
data — prefer a rented raw GPU where you delete the volume after, over hosted
fine-tuning APIs that retain datasets.

## RunPod steps (the recommended path)

1. runpod.io → Deploy → RTX 4090, template "PyTorch 2.x". ~$0.40/h community cloud.
2. In the pod's terminal:
   ```bash
   git clone <this-repo> && cd Arvind-Lora
   pip install unsloth
   python3 training/prepare_training_data.py
   python3 training/train_unsloth.py --model llama --epochs 1   # smoke test, ~20 min
   python3 training/train_unsloth.py --model gemma              # real run
   ```
3. Download `training/lora_out/gguf/` (one ~7 GB file) and the `adapter/` folder,
   then **terminate the pod and delete the volume**.
4. Locally: `ollama create arvind -f Modelfile` (see PLAN.md Phase 5).

## Colab free path ($0)

Runtime → T4 GPU. Upload the repo (or clone it), same commands but
`--model llama`. Expect a few hours; keep the tab alive.

## What to look at during the run

- `eval_loss` should still be falling at epoch 1's end; if it climbs during epoch 2,
  stop — the adapter is starting to memorize.
- Every checkpoint, generate a few samples by hand (a casual NO brief, a formal
  complaint, an English booking) and score them against
  `data/style-profile.md` → "Anti-patterns".

## After training: the eval that matters

PLAN.md Phase 4 — blind A/B with people who know Arvind. Don't skip it; val loss
can't measure "indistinguishable".
