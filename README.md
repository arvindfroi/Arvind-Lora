# Arvind-Lora

A personal writing-style dataset and fine-tuning project. Goal: a LoRA-tuned LLM that
writes **indistinguishably from Arvind Frøiland** — in both Norwegian and English, and in
the right *register* for the situation (casual email, formal complaint, academic prose,
study notes, open-source community mail).

> ⚠️ **Privacy: keep this repository PRIVATE.** `data/` contains verbatim personal
> emails and documents. Third-party full names, phone numbers and other sensitive
> details have been redacted with `[...]` placeholders, but the content is still personal.

## Repository layout

```
data/
  corpus.jsonl        # The curated corpus. One sample per line, with metadata.
  style-profile.md    # Analysis of Arvind's voice, per language and register.
  curation-notes.md   # Where the data came from, what was excluded and why,
                      # and what to collect next.
  EXPORTS.md          # How to export Snapchat/iMessage/WhatsApp/Instagram data
                      # (no APIs exist for these — official exports only).
  chat/               # Output of ingest_chats.py — review before committing;
                      # contains quoted context from chat partners.
scripts/
  build_dataset.py    # corpus.jsonl -> chat-format train/val JSONL for SFT.
  ingest_chats.py     # chat exports -> training samples (partners anonymized).
PLAN.md               # The full fine-tuning plan (model options, LoRA recipe,
                      # evaluation, serving).
```

## Quick start

```bash
# Inspect the corpus
python3 scripts/build_dataset.py --stats

# Build training data (gold + silver tiers, 90/10 split)
python3 scripts/build_dataset.py --tiers gold,silver --out out/
```

Then follow `PLAN.md` for training.

## The one rule of this dataset

**Only text Arvind actually wrote goes in the gold tier.** AI-assisted drafts (he uses
Claude as a drafting tool for some polished emails) are tiered `silver`/`bronze` and
flagged, because training on them teaches the model to sound like an assistant, not
like Arvind. The fingerprint lives in the spontaneous, unedited writing — typos,
dialect, smileys and all.
