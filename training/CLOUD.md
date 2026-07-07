# Cloud training (Qwythos LoRA) — the ~$2, ~1–2h path

Local Qwythos training is impractical on the RTX 5080 Laptop (Blackwell kernels
untuned → ~4 min/step, ~40h). A rented datacenter GPU runs the *same* scripts in
~1–2h. Everything here is already in the repo; `training/cloud_setup.sh` does it all.

## Steps

1. **Get a GitHub token.** github.com/settings/tokens → fine-grained PAT →
   read access to `arvindfroi/Arvind-Lora`. Copy it.

2. **Rent a pod.** [runpod.io](https://runpod.io) (or Vast/Lambda) → Deploy →
   **1× RTX 4090 24GB**, template **"PyTorch 2.x"**, ~50GB disk. Community cloud
   is ~$0.35–0.70/h. Open the pod's web terminal.

3. **Run one block:**
   ```bash
   export GITHUB_TOKEN=github_pat_xxxxx
   cd /workspace && git clone https://$GITHUB_TOKEN@github.com/arvindfroi/Arvind-Lora
   bash Arvind-Lora/training/cloud_setup.sh
   ```
   It clones, installs (incl. flash-attn so packing is on), downloads the 19GB
   model, builds the balanced dataset, and trains 2 epochs. Watch `loss` fall.

4. **Download the adapter, then destroy the pod:**
   ```bash
   cd /workspace/Arvind-Lora/training/lora_out_qwythos && zip -r adapter.zip adapter
   ```
   Download `adapter.zip` via the pod file browser. **Then terminate the pod and
   delete the volume** — the corpus is private (PLAN.md's rule).

5. **Use it locally.** Unzip to `training/lora_out_qwythos/adapter`. Serve with
   the base model + this LoRA (vLLM/transformers), or merge + export GGUF for
   Ollama per PLAN.md Phase 5.

## Notes
- Cost: budget **$2–5** for 2–3 runs. A single 2-epoch run is well under an hour of GPU.
- Privacy: the private repo (with your emails/chats) is cloned to the pod. Rent a
  raw GPU you control and delete the volume after — do not use a hosted
  fine-tuning API that retains data.
- The scripts (`train_qwythos.py`, `prepare_training_data.py`) are identical to
  what ran locally; only `--packing --attn flash_attention_2` is added, which the
  cloud GPU supports.
