#!/usr/bin/env python3
"""Ingest chat exports (Snapchat, iMessage, WhatsApp, Instagram/Messenger, Discord)
into chat-format SFT samples that match scripts/build_dataset.py output.

The model can't be given live access to these platforms, so this consumes their
official data exports instead. See data/EXPORTS.md for how to obtain each export.

Output format per line (same as out/train.jsonl from build_dataset.py):
  {"id": ..., "messages": [{"role":"system",...},{"role":"user",...},{"role":"assistant",...}]}

- user turn   = the preceding conversation window, partners anonymized
- assistant   = Arvind's actual reply (consecutive messages merged)

Usage:
  python3 scripts/ingest_chats.py --platform snapchat  --input mydata/json/chat_history.json --out data/chat/
  python3 scripts/ingest_chats.py --platform whatsapp  --input export/_chat.txt              --out data/chat/
  python3 scripts/ingest_chats.py --platform messenger --input messages/inbox/               --out data/chat/
  python3 scripts/ingest_chats.py --platform imessage  --input imessage-export/              --out data/chat/
"""

import argparse
import json
import re
from datetime import datetime, timedelta
from pathlib import Path

ME_DEFAULT = "arvind,arvind frøiland,arvind froiland,arvindfroi,me,meg,du"

SYSTEM_CHAT = (
    "Du er Arvind Frøiland i en privat chat. Du svarer nøyaktig slik Arvind chatter - "
    "korte meldinger, hans slang, dialekt, skrivefeil og tegnsetting.\n"
    "Språk: {lang}. Register: chat-{platform}."
)

MEDIA_PLACEHOLDERS = re.compile(
    r"^(<Media omitted>|image omitted|video omitted|audio omitted|sticker omitted|"
    r"GIF omitted|‎?bilde utelatt|‎?video utelatt|Media|MEDIA|"
    r"You sent an attachment.*|Sendte et vedlegg.*)$",
    re.IGNORECASE,
)

NO_HINTS = {"og", "jeg", "det", "ikke", "på", "er", "som", "til", "med", "har",
            "deg", "meg", "når", "skal", "kan", "hva", "hvis", "også", "takk",
            "eg", "då", "ka", "kje", "å"}
EN_HINTS = {"the", "and", "you", "that", "have", "this", "with", "for", "was",
            "are", "what", "when", "your", "just", "like", "yeah", "okay"}


def detect_lang(text):
    words = re.findall(r"[a-zæøåA-ZÆØÅ']+", text.lower())
    no = sum(w in NO_HINTS for w in words)
    en = sum(w in EN_HINTS for w in words)
    if no > en:
        return "no"
    if en > no:
        return "en"
    return "no" if re.search(r"[æøå]", text) else "en"


# --- Platform parsers. Each yields (conversation_id, datetime, sender, text). ---

def parse_snapchat(path):
    """Snapchat My Data: json/chat_history.json.
    Handles both the {"Received Chat History": [...], "Sent Chat History": [...]}
    layout and the newer {friend_username: [messages]} layout."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))

    def emit(msg, direction_hint=None):
        text = msg.get("Content") or msg.get("Text") or ""
        if msg.get("Media Type", "TEXT").upper() != "TEXT" or not text:
            return None
        created = msg.get("Created", "")
        try:
            ts = datetime.strptime(created.replace(" UTC", ""), "%Y-%m-%d %H:%M:%S")
        except ValueError:
            return None
        other = msg.get("From") or msg.get("To") or "unknown"
        if direction_hint == "sent":
            sender = "me"
        elif direction_hint == "received":
            sender = other
        else:
            sender = "me" if msg.get("IsSender") else other
        return (other, ts, sender, text)

    if isinstance(data, dict) and "Sent Chat History" in data:
        for msg in data.get("Received Chat History", []):
            row = emit(msg, "received")
            if row:
                yield row
        for msg in data.get("Sent Chat History", []):
            row = emit(msg, "sent")
            if row:
                yield row
    elif isinstance(data, dict):
        for friend, msgs in data.items():
            for msg in msgs:
                row = emit(msg)
                if row:
                    yield (friend, *row[1:])


WHATSAPP_RE = re.compile(
    r"^\[?(\d{1,2}[./]\d{1,2}[./]\d{2,4}),? (\d{1,2}:\d{2}(?::\d{2})?)\]?\s*[-–]?\s*([^:]+): (.*)$"
)


def parse_whatsapp(path):
    """One exported _chat.txt (Settings -> Chats -> Export Chat, without media)."""
    conv = Path(path).stem
    current = None
    for raw in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip("‎‏ ")
        m = WHATSAPP_RE.match(line)
        if m:
            date_s, time_s, sender, text = m.groups()
            for fmt in ("%d.%m.%Y %H:%M:%S", "%d.%m.%Y %H:%M", "%d/%m/%Y %H:%M:%S",
                        "%d/%m/%Y %H:%M", "%d.%m.%y %H:%M:%S", "%d.%m.%y %H:%M",
                        "%d/%m/%y %H:%M"):
                try:
                    ts = datetime.strptime(f"{date_s} {time_s}", fmt)
                    break
                except ValueError:
                    ts = None
            if ts is None:
                current = None
                continue
            current = [conv, ts, sender.strip(), text]
            yield tuple(current)
        elif current and line:
            # continuation line of a multi-line message
            current[3] = line
            yield (current[0], current[1], current[2], line)


def parse_messenger(path):
    """Meta 'Download your information' JSON: a folder like messages/inbox/<thread>/
    with message_1.json etc. Pass either one thread folder or the whole inbox/."""
    root = Path(path)
    files = sorted(root.rglob("message_*.json"))
    for f in files:
        data = json.loads(f.read_text(encoding="utf-8"))
        conv = data.get("title") or f.parent.name
        for msg in data.get("messages", []):
            text = msg.get("content")
            if not text:
                continue
            # Meta exports encode UTF-8 as latin-1 escapes; undo it.
            try:
                text = text.encode("latin-1").decode("utf-8")
            except (UnicodeEncodeError, UnicodeDecodeError):
                pass
            ts = datetime.fromtimestamp(msg["timestamp_ms"] / 1000)
            sender = msg.get("sender_name", "unknown")
            try:
                sender = sender.encode("latin-1").decode("utf-8")
            except (UnicodeEncodeError, UnicodeDecodeError):
                pass
            yield (conv, ts, sender, text)


IMESSAGE_DATE_RE = re.compile(r"^[A-Z][a-z]{2} \d{1,2}, \d{4}\s+\d{1,2}:\d{2}:\d{2} [AP]M")


def parse_imessage(path):
    """Output of `imessage-exporter -f txt`: one .txt per conversation, blocks of
    [date line, sender line, message lines..., blank]."""
    for f in sorted(Path(path).glob("*.txt")):
        conv = f.stem
        lines = f.read_text(encoding="utf-8", errors="replace").splitlines()
        i = 0
        while i < len(lines):
            if IMESSAGE_DATE_RE.match(lines[i].strip()):
                date_s = lines[i].strip().split(" (")[0]
                try:
                    ts = datetime.strptime(date_s, "%b %d, %Y %I:%M:%S %p")
                except ValueError:
                    i += 1
                    continue
                if i + 1 >= len(lines):
                    break
                sender = lines[i + 1].strip()
                body = []
                i += 2
                while i < len(lines) and lines[i].strip() and not IMESSAGE_DATE_RE.match(lines[i].strip()):
                    body.append(lines[i].strip())
                    i += 1
                if body:
                    yield (conv, ts, sender, "\n".join(body))
            else:
                i += 1


PARSERS = {
    "snapchat": parse_snapchat,
    "whatsapp": parse_whatsapp,
    "messenger": parse_messenger,
    "instagram": parse_messenger,  # same Meta export format
    "imessage": parse_imessage,
}


# --- Sample construction ---

def is_me(sender, me_names):
    s = sender.strip().lower()
    return s == "me" or s in me_names


def clean(text):
    text = text.strip()
    if not text or MEDIA_PLACEHOLDERS.match(text):
        return None
    if re.fullmatch(r"https?://\S+", text):
        return None
    return text


def build_samples(rows, me_names, platform, context_window=6, merge_minutes=3,
                  min_reply_chars=3):
    """Group rows per conversation, merge consecutive same-sender messages,
    emit (context, reply) pairs for every run of my messages that has context."""
    convs = {}
    for conv, ts, sender, text in rows:
        text = clean(text)
        if text:
            convs.setdefault(conv, []).append((ts, sender, text))

    samples = []
    partner_ids = {}
    for conv, msgs in convs.items():
        msgs.sort(key=lambda r: r[0])
        # merge runs
        merged = []
        for ts, sender, text in msgs:
            if merged and merged[-1][1] == sender and \
                    ts - merged[-1][0] <= timedelta(minutes=merge_minutes):
                merged[-1] = (ts, sender, merged[-1][2] + "\n" + text)
            else:
                merged.append((ts, sender, text))

        pid = partner_ids.setdefault(conv, f"Partner{len(partner_ids) + 1}")
        for idx, (ts, sender, text) in enumerate(merged):
            if not is_me(sender, me_names) or len(text) < min_reply_chars or idx == 0:
                continue
            ctx_rows = merged[max(0, idx - context_window):idx]
            ctx = "\n".join(
                f"{'Meg' if is_me(s, me_names) else pid}: {t}" for _, s, t in ctx_rows
            )
            lang = detect_lang(text)
            samples.append({
                "id": f"{platform}-{pid}-{ts:%Y%m%d%H%M%S}-{idx}",
                "messages": [
                    {"role": "system",
                     "content": SYSTEM_CHAT.format(lang="norsk" if lang == "no" else "engelsk",
                                                   platform=platform)},
                    {"role": "user", "content": ctx},
                    {"role": "assistant", "content": text},
                ],
            })
    return samples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--platform", required=True, choices=sorted(PARSERS))
    ap.add_argument("--input", required=True, help="export file or folder (see data/EXPORTS.md)")
    ap.add_argument("--out", default="data/chat", help="output directory")
    ap.add_argument("--me", default=ME_DEFAULT,
                    help="comma-separated names/handles that identify your own messages")
    ap.add_argument("--context-window", type=int, default=6)
    args = ap.parse_args()

    me_names = {n.strip().lower() for n in args.me.split(",") if n.strip()}
    rows = PARSERS[args.platform](args.input)
    samples = build_samples(rows, me_names, args.platform,
                            context_window=args.context_window)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    dest = out / f"{args.platform}.jsonl"
    with open(dest, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"wrote {len(samples)} samples -> {dest}")
    if not samples:
        print("0 samples: check --me names match how the export labels your messages "
              "(run with a single conversation and inspect the raw file).")


if __name__ == "__main__":
    main()
