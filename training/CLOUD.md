# Cloud training (Qwythos LoRA) — the ~$1–2, ~2–3h path

Local Qwythos training is impractical on the RTX 5080 Laptop (Blackwell sm_120 has no
tuned flash-linear-attention kernels → ~4 min/step, ~40h). A rented datacenter GPU runs
the *same* scripts in a couple of hours. `training/cloud_setup.sh` does all of it.

## Provider: RunPod Secure Cloud, 1× A40 48GB, EU region

| | |
|---|---|
| **Provider** | RunPod — **Secure Cloud**, not Community Cloud |
| **GPU** | 1× A40 48GB (~$0.44/h) |
| **Region** | any EU (EU-RO-1 / EU-SE-1 / EU-CZ-1) |
| **Template** | "PyTorch 2.x", ~60GB disk |
| **Run cost** | ~$1–2 for a full 2-epoch run |

Why this and not the alternatives:

- **Secure Cloud over Community Cloud / Vast.ai.** Community and Vast pods run on
  *other people's machines*. This corpus is verbatim personal email and private chats
  involving named third parties. Secure Cloud is RunPod's own SOC2 datacenters. The
  ~$0.10–0.40/h premium is the entire cost of not putting your friends' Snapchat
  messages on a stranger's GPU.
- **EU region.** Same data, now also a GDPR question — most of it is personal data about
  Norwegian citizens who never consented to a US transfer. An EU region moots it.
- **A40 48GB over RTX 4090 24GB.** The A40 is *cheaper* ($0.44 vs $0.69/h) **and** has
  double the VRAM. 48GB lets us drop gradient checkpointing and run batch 4 instead of
  batch 1 — so it is also faster per dollar. `cloud_setup.sh` detects VRAM and picks the
  batch size automatically.
- **Not a hosted fine-tuning API** (OpenAI/Together/Fireworks). Those retain your
  training data. Rent a raw GPU you control and delete the volume. This is PLAN.md's rule.

L40 48GB or A6000 48GB are fine substitutes if A40 is out of stock.

## Steps

1. **Get a GitHub token.** github.com/settings/tokens → fine-grained PAT →
   read access to `arvindfroi/Arvind-Lora`. Copy it.

2. **Rent the pod.** [runpod.io](https://runpod.io) → Deploy → **Secure Cloud** →
   filter to an **EU** region → **A40 48GB** → template **"PyTorch 2.x"** → ~60GB disk.
   Open the pod's web terminal.

3. **Run one block:**
   ```bash
   export GITHUB_TOKEN=github_pat_xxxxx
   cd /workspace && git clone https://$GITHUB_TOKEN@github.com/arvindfroi/Arvind-Lora
   bash Arvind-Lora/training/cloud_setup.sh
   ```
   It clones, installs (incl. flash-attn so packing is on), downloads the 19GB model,
   builds the balanced dataset, and trains 2 epochs. Watch `loss` fall on the progress bar.

4. **Download the adapter, then destroy the pod:**
   ```bash
   cd /workspace/Arvind-Lora/training/lora_out_qwythos && zip -r adapter.zip adapter
   ```
   Download `adapter.zip` via the pod file browser. **Then terminate the pod and delete
   the volume.** The adapter is a few hundred MB; the corpus must not outlive the run.

5. **Use it locally.** Unzip to `training/lora_out_qwythos/adapter`. Serve with the base
   model + this LoRA (vLLM/transformers), or merge + export GGUF for Ollama per PLAN.md
   Phase 5.

## Notes

- Budget **$5** to leave room for 2–3 runs if the first needs a hyperparameter tweak.
- The scripts are identical to what ran locally; the cloud adds `--packing
  --attn flash_attention_2` and a larger batch, which the datacenter GPU supports.
- Sequence packing **requires** FlashAttention on this hybrid arch. Without it, samples
  cross-contaminate — fatal for style training. `cloud_setup.sh` only enables packing
  when `flash_attn` actually imported.

## Dataset the pod will build

`prepare_training_data.py --max-chat-per-file 3500 --cap chatgpt=2500` →
**8,312 train / 552 val samples (~1.82M tokens)**. By *target* tokens (the assistant turn,
i.e. what the model learns to produce): curated gold 56%, ChatGPT 18%, Snapchat 11%,
Claude 10%, Messenger 2%, Discord 2%.

ChatGPT is capped tighter than the others because 5,082 raw samples of Arvind prompting
an AI would otherwise crowd out the Snapchat dialect core. Only Arvind's own turns are
ever training targets — AI replies appear solely as truncated context.
