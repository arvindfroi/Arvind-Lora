#!/usr/bin/env python3
"""Build the final train/val JSONL for the LoRA run.

Merges:
  - data/corpus.jsonl   (curated email/exam/doc samples; brief -> text)
  - data/chat/*.jsonl   (already chat-format: context -> reply)

Because chat outnumbers everything ~60:1, non-chat registers are upsampled in
the train split (default 4x) so the formal voices don't drown. Val stays
un-upsampled and stratified.

Usage:
  python3 training/prepare_training_data.py            # writes training/out/
  python3 training/prepare_training_data.py --upsample 4 --chat-val-ratio 0.02
"""

import argparse
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

SYSTEM_TEMPLATE = (
    "Du er Arvind Frøiland. Du skriver nøyaktig slik Arvind skriver - med hans "
    "ordvalg, tegnsetting, skrivefeil og tone. Du polerer aldri teksten utover "
    "slik han faktisk skriver.\n"
    "Språk: {lang}. Register: {register}."
)
LANG_NAMES = {"no": "norsk", "en": "engelsk"}


def corpus_to_chat(entry):
    return {
        "id": entry["id"],
        "messages": [
            {"role": "system", "content": SYSTEM_TEMPLATE.format(
                lang=LANG_NAMES.get(entry["lang"], entry["lang"]),
                register=entry["register"])},
            {"role": "user", "content": entry["brief"]},
            {"role": "assistant", "content": entry["text"]},
        ],
    }


def est_tokens(sample):
    # Norwegian/English mixed text lands around ~3.3 chars/token on modern BPEs.
    chars = sum(len(m["content"]) for m in sample["messages"])
    return int(chars / 3.3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--upsample", type=int, default=4,
                    help="repeat factor for curated (non-chat) samples in train")
    ap.add_argument("--tiers", default="gold,silver")
    ap.add_argument("--chat-val-ratio", type=float, default=0.02)
    ap.add_argument("--corpus-val-ratio", type=float, default=0.10)
    ap.add_argument("--max-chat-per-file", type=int, default=0,
                    help="cap train samples kept per chat file (0 = keep all). "
                         "Trims the redundant Snapchat tail; keeps small files "
                         "(Messenger) whole. Improves register balance and "
                         "training time without touching val.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=str(ROOT / "training" / "out"))
    args = ap.parse_args()
    random.seed(args.seed)

    tiers = set(args.tiers.split(","))

    # curated corpus
    curated = []
    for line in open(ROOT / "data" / "corpus.jsonl", encoding="utf-8"):
        e = json.loads(line)
        if e["tier"] in tiers and e.get("complete", True):
            curated.append(corpus_to_chat(e))
    random.shuffle(curated)
    n_val = max(1, int(len(curated) * args.corpus_val_ratio))
    corpus_val, corpus_train = curated[:n_val], curated[n_val:]

    # chat data
    chat_train, chat_val = [], []
    for f in sorted((ROOT / "data" / "chat").glob("*.jsonl")):
        rows = [json.loads(l) for l in open(f, encoding="utf-8")]
        random.shuffle(rows)
        k = int(len(rows) * args.chat_val_ratio)
        chat_val.extend(rows[:k])
        keep = rows[k:]
        if args.max_chat_per_file and len(keep) > args.max_chat_per_file:
            keep = keep[:args.max_chat_per_file]   # already shuffled above
        chat_train.extend(keep)

    train = corpus_train * args.upsample + chat_train
    random.shuffle(train)
    val = corpus_val + chat_val

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, split in (("train", train), ("val", val)):
        with open(out / f"{name}.jsonl", "w", encoding="utf-8") as f:
            for s in split:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")

    t_train = sum(est_tokens(s) for s in train)
    t_val = sum(est_tokens(s) for s in val)
    print(f"train: {len(train):6d} samples (~{t_train/1e6:.2f}M tokens) "
          f"[curated {len(corpus_train)}x{args.upsample} + chat {len(chat_train)}]")
    print(f"val:   {len(val):6d} samples (~{t_val/1e6:.2f}M tokens)")
    print(f"-> {out}/train.jsonl, {out}/val.jsonl")


if __name__ == "__main__":
    main()
