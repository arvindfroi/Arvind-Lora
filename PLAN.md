# Fine-tuning plan: an LLM that writes exactly like Arvind

Goal: a model that, given a situation ("reply to the landlord about the noise
complaint", "send en purring til eksamenskontoret"), produces text a reader who knows
Arvind cannot distinguish from the real thing — in Norwegian **and** English, in the
right register for the content.

## Phase 0 — Grow the corpus (the thing that actually decides success)

Current state: **5,546 training samples** — 96 curated email/doc samples
(`data/corpus.jsonl`, ~3 900 words) plus **5,450 Snapchat chat samples**
(`data/chat/snapchat.jsonl`, ~54 800 words of Arvind-text, 2024→2026, both
languages, full dialect). This crossed the threshold where a style LoRA becomes
genuinely viable; chat is now the dominant register, which matches how he
actually writes day-to-day.

Remaining collection (see `data/curation-notes.md`):

1. iMessage via Mac `imessage-exporter` (pipeline ready).
2. Apple data request (in flight): Notes + iCloud Drive docs.
3. Solo school essays and the bachelor draft for the academic register.
4. Re-fetch the ~24 truncated email samples marked `"complete": false`.

Mix guidance now that chat dominates: email/formal registers are outnumbered 57:1.
Either upsample the email corpus ~3–5× in training, or accept a chat-leaning clone
and rely on the register tag to switch modes — decide after the first eval run.

Hard rule carried through every phase: **AI-drafted text stays out of the gold tier.**
Arvind uses Claude to draft polished mail; training on that produces an assistant
impersonating Arvind rather than Arvind. Silver tier is capped at ~20% of the mix.

## Phase 1 — Dataset construction

`scripts/build_dataset.py` already produces chat-format SFT data:

- **system**: fixed persona line + `lang` + `register` tags. Tagging registers
  explicitly is what lets one model hold multiple voices without blending them.
- **user**: a natural-language brief describing the situation (already hand-written
  for every corpus entry).
- **assistant**: the verbatim text.

When the corpus grows, generate briefs automatically: feed each new sample to a
strong LLM and ask it to write the *instruction that would have produced this text*
(reverse instruction / self-instruct). Spot-check 1 in 10.

Augmentations worth adding at >300 samples:

- **Continuation pairs**: first half of a text as prompt → second half as target
  (teaches the voice without needing briefs).
- **Anti-style DPO pairs**: for each brief, keep Arvind's real text as *chosen* and a
  generic polished LLM answer as *rejected*. This is the cheapest, most effective trick
  for killing assistant-speak — exactly the "indistinguishable" requirement.

## Phase 2 — Model choice (decision deferred, as requested)

Requirements: strong Norwegian Bokmål, open weights, runnable locally, LoRA-friendly.

| Option | Size | Norwegian quality | Notes |
|---|---|---|---|
| **Gemma 3 12B** (recommended default) | 12B | very good | Best Scandinavian quality per GB in the open-weights field; QLoRA fits in 16 GB VRAM |
| Llama 3.1 8B | 8B | decent | Cheapest iteration loop; good first prototype |
| Mistral Nemo 12B | 12B | good | Strong multilingual tokenizer |
| NorMistral / NB-Llama (NorwAI, Nasjonalbiblioteket) | 7–8B | native-focused | Trained on Norwegian corpora; weaker instruction-following — pair with more SFT data |
| Qwen 3 14B | 14B | good | Strong generalist alternative |

Recommendation: prototype on Llama 3.1 8B (fast, cheap), do the real runs on
Gemma 3 12B, and A/B against NorMistral if the Norwegian ever feels off.

Note: hosted fine-tuning APIs (OpenAI, Together, etc.) also work, but local
open-weights + LoRA keeps the personal corpus on your own hardware — given how private
this data is, that's the sane default.

## Phase 3 — Training recipe (QLoRA)

Tooling: **Unsloth** (single-GPU friendly, free Colab works) or **Axolotl**.

```yaml
# Axolotl-style sketch
base_model: google/gemma-3-12b-it
load_in_4bit: true            # QLoRA
adapter: lora
lora_r: 32
lora_alpha: 64
lora_dropout: 0.05
lora_target_modules: all-linear
sequence_len: 2048
micro_batch_size: 2
gradient_accumulation_steps: 8   # effective batch 16
learning_rate: 1.5e-4
lr_scheduler: cosine
warmup_ratio: 0.05
num_epochs: 4                    # small data: watch eval loss from epoch 2
val_set: out/val.jsonl
```

- Small-data regime: overfitting shows up as the model parroting whole training emails.
  Early-stop on val loss and eyeball generations every epoch.
- Hardware: any 16–24 GB GPU (RTX 4090/3090, or rented on RunPod/Vast ~$0.30/h).
  A full run at this data size is minutes, not hours.
- Stage 2 (once DPO pairs exist): one epoch of DPO/ORPO on chosen=Arvind vs
  rejected=generic-LLM. β≈0.1. This is what closes the "sounds almost like me" gap.

## Phase 4 — Evaluation: "indistinguishable" made measurable

1. **Blind A/B (the real test).** 20 briefs, model + Arvind both write answers, people
   who know him guess which is which. Success = guessers at ~50% (coin-flip).
2. **Register checklist** from `data/style-profile.md`: does casual NO contain
   smileys/comma splices? Does formal NO close with "Med vennlig hilsen"? Does casual
   EN keep the L2 fingerprints? Score each generation against the anti-patterns list.
3. **Stylometry.** Burrows' Delta / function-word frequency between generated text and
   held-out real text, compared against a generic-LLM baseline. Cheap and objective.
4. **Held-out val loss** per register (the script's stratified split enables this).

Iterate: whichever register fails, collect more of it (Phase 0) and retrain.

## Phase 5 — Serving

- Merge LoRA or keep adapter separate; export GGUF; run in **Ollama**:
  `ollama create arvind -f Modelfile` with the same system-prompt template the
  training used (important — the register tags are part of the contract).
- Front it with anything (Open WebUI, a Shortcuts action on the phone, a small API).
  At inference time, pick `lang` + `register` per task, exactly as trained.

## Ethics & safety notes

- Keep this repo and all trained weights **private** — the corpus is personal data,
  and the model itself memorizes it.
- The clone writes *as* Arvind for Arvind. Where a community requires AI disclosure
  (Fedora, PPSSPP — he already discloses today), keep disclosing; the model even
  learned his disclosure habit from the silver tier.
- Don't let the model send anything autonomously without review until Phase 4 passes.
