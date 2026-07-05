#!/usr/bin/env python3
"""Convert data/corpus.jsonl into chat-format SFT data (train/val JSONL).

Each corpus entry becomes one training example:
  system: persona + language/register tags
  user:   the situational brief (what needs to be written)
  assistant: Arvind's actual text, verbatim

Usage:
  python3 scripts/build_dataset.py --stats
  python3 scripts/build_dataset.py --tiers gold,silver --out out/ --val-ratio 0.1
"""

import argparse
import json
import random
from collections import Counter
from pathlib import Path

CORPUS = Path(__file__).resolve().parent.parent / "data" / "corpus.jsonl"

SYSTEM_TEMPLATE = (
    "Du er Arvind Frøiland. Du skriver nøyaktig slik Arvind skriver - med hans "
    "ordvalg, tegnsetting, skrivefeil og tone. Du polerer aldri teksten utover "
    "slik han faktisk skriver.\n"
    "Språk: {lang}. Register: {register}."
)

LANG_NAMES = {"no": "norsk", "en": "engelsk", "mixed": "norsk/engelsk blandet"}


def load_corpus():
    entries = []
    with open(CORPUS, encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise SystemExit(f"corpus.jsonl line {line_no}: invalid JSON ({e})")
    return entries


def to_chat(entry):
    system = SYSTEM_TEMPLATE.format(
        lang=LANG_NAMES.get(entry["lang"], entry["lang"]),
        register=entry["register"],
    )
    return {
        "id": entry["id"],
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": entry["brief"]},
            {"role": "assistant", "content": entry["text"]},
        ],
    }


def print_stats(entries):
    print(f"total samples: {len(entries)}")
    for field in ("lang", "register", "tier", "complete"):
        counts = Counter(str(e.get(field)) for e in entries)
        pretty = ", ".join(f"{k}={v}" for k, v in counts.most_common())
        print(f"  by {field}: {pretty}")
    words = sum(len(e["text"].split()) for e in entries)
    print(f"  total words of Arvind-text: {words}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stats", action="store_true", help="print corpus stats and exit")
    ap.add_argument("--tiers", default="gold,silver", help="comma-separated tiers to include")
    ap.add_argument("--include-incomplete", action="store_true",
                    help="include samples whose text was truncated at collection time")
    ap.add_argument("--out", default="out", help="output directory")
    ap.add_argument("--val-ratio", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    entries = load_corpus()
    if args.stats:
        print_stats(entries)
        return

    tiers = set(args.tiers.split(","))
    selected = [e for e in entries if e["tier"] in tiers]
    if not args.include_incomplete:
        selected = [e for e in selected if e.get("complete", True)]

    # Stratify the split by register so small registers keep train coverage.
    random.seed(args.seed)
    by_register = {}
    for e in selected:
        by_register.setdefault(e["register"], []).append(e)

    train, val = [], []
    for register, group in sorted(by_register.items()):
        random.shuffle(group)
        n_val = int(len(group) * args.val_ratio)
        val.extend(group[:n_val])
        train.extend(group[n_val:])
    random.shuffle(train)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, split in (("train", train), ("val", val)):
        path = out / f"{name}.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for e in split:
                f.write(json.dumps(to_chat(e), ensure_ascii=False) + "\n")
        print(f"wrote {len(split):4d} samples -> {path}")

    print_stats(selected)


if __name__ == "__main__":
    main()
