#!/usr/bin/env python3
"""Ingest the newly-exported sources (Claude, Discord, academic essays) into the
training corpus — keeping ONLY Arvind's own text.

  - Claude export  (conversations.json)  -> data/chat/claude.jsonl
      Only the *human* turns (Arvind's messages). Per the repo rule, the
      assistant/AI turns are dropped; we train on his replies, with the
      preceding AI message kept as context so the target has something to answer.
  - Discord export (package/Messages/*)  -> data/chat/discord.jsonl
      Discord only exports YOUR OWN messages, so every line is his. No partner
      text exists, so these are unconditioned style samples.
  - Academic essays (.md, allow-listed)  -> appended to data/corpus.jsonl (gold)

Deliberately EXCLUDED: AI-generated .md (compass_artifact*, deep-research*,
openclaw*, smash-*, chat-*) and the AI turns in the Claude export.

  python scripts/ingest_new_sources.py --downloads ~/Downloads          # write
  python scripts/ingest_new_sources.py --downloads ~/Downloads --dry-run
"""

import argparse
import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

SYSTEM_CHAT = (
    "Du er Arvind Frøiland i en privat chat. Du svarer nøyaktig slik Arvind "
    "chatter - korte meldinger, hans slang, dialekt, skrivefeil og tegnsetting.\n"
    "Språk: {lang}. Register: {register}."
)
SYSTEM_AICHAT = (
    "Du er Arvind Frøiland. Du skriver meldingene dine til en AI-assistent "
    "nøyaktig slik Arvind skriver dem - hans ordvalg, tone og tegnsetting.\n"
    "Språk: {lang}. Register: {register}."
)

NO_HINTS = {"og", "jeg", "det", "ikke", "på", "er", "som", "til", "med", "har",
            "deg", "meg", "når", "skal", "kan", "hva", "hvis", "også", "takk",
            "eg", "då", "ka", "kje", "å", "jo", "ein", "eg", "kanskje"}
EN_HINTS = {"the", "and", "you", "that", "have", "this", "with", "for", "was",
            "are", "what", "when", "your", "just", "like", "yeah", "okay", "can"}


def detect_lang(text):
    words = re.findall(r"[a-zæøåA-ZÆØÅ']+", text.lower())
    no = sum(w in NO_HINTS for w in words)
    en = sum(w in EN_HINTS for w in words)
    if no > en:
        return "no"
    if en > no:
        return "en"
    return "no" if re.search(r"[æøå]", text) else "en"


def lang_name(code):
    return {"no": "norsk", "en": "engelsk"}.get(code, code)


CODE_HINTS = re.compile(r"```|def |function |import |const |</|/>|SELECT |{\n|};|\bpip install\b")


def is_his_prose(text, lo=12, hi=2000):
    """Keep genuine short-form human text; drop pastes, code, logs, URL-only."""
    t = (text or "").strip()
    if not (lo <= len(t) <= hi):
        return False
    if t.startswith(("http://", "https://", "!", "/")):  # link-only / bot cmd
        return False
    if CODE_HINTS.search(t):
        return False
    letters = sum(c.isalpha() for c in t)
    if letters < 0.5 * len(t):          # too many symbols -> not prose
        return False
    if t.count("\n") > 12:              # long pasted block
        return False
    return True


# ---------------------------------------------------------------- Claude
def parse_claude(conv_path):
    convs = json.load(open(conv_path, encoding="utf-8"))
    out = []
    for c in convs:
        msgs = c.get("chat_messages", [])
        last_ai = ""
        for m in msgs:
            sender = m.get("sender")
            text = (m.get("text") or "").strip()
            if sender == "assistant":
                last_ai = text
                continue
            if sender != "human":
                continue
            if not is_his_prose(text):
                last_ai = ""       # reset context after a dropped/huge turn
                continue
            lang = detect_lang(text)
            if last_ai:
                ctx = last_ai if len(last_ai) <= 700 else last_ai[:700] + " […]"
                user = f"Assistenten svarte:\n{ctx}"
            else:
                user = f"(Start på en samtale med en AI-assistent: «{c.get('name','')}»)"
            out.append({
                "id": f"claude-{m.get('uuid','')[:8]}",
                "messages": [
                    {"role": "system", "content": SYSTEM_AICHAT.format(
                        lang=lang_name(lang), register="chat-ai")},
                    {"role": "user", "content": user},
                    {"role": "assistant", "content": text},
                ],
            })
            last_ai = ""
    return out


# ---------------------------------------------------------------- Discord
DISCORD_BRIEFS = [
    "(Skriv en uformell Discord-melding slik du pleier.)",
    "(Send en kort melding i en Discord-kanal.)",
    "(Skriv noe i chatten slik du vanligvis gjør.)",
]


def parse_discord(messages_root):
    out = []
    n = 0
    for chan_dir in sorted(Path(messages_root).glob("c*")):
        mfile = chan_dir / "messages.json"
        if not mfile.exists():
            continue
        rows = json.load(open(mfile, encoding="utf-8"))
        # export is newest-first; go chronological
        rows = list(reversed(rows))
        # merge consecutive messages into small runs
        run = []
        for r in rows:
            c = (r.get("Contents") or "").strip()
            if is_his_prose(c, lo=2):
                run.append(c)
            else:
                if run:
                    _emit_discord(run, out, n); n += 1; run = []
        if run:
            _emit_discord(run, out, n); n += 1
    return out


def _emit_discord(run, out, n):
    text = "\n".join(run[:8])          # cap a run at 8 lines
    if len(text) < 3:
        return
    lang = detect_lang(text)
    out.append({
        "id": f"discord-{n:05d}",
        "messages": [
            {"role": "system", "content": SYSTEM_CHAT.format(
                lang=lang_name(lang), register="chat-discord")},
            {"role": "user", "content": DISCORD_BRIEFS[n % len(DISCORD_BRIEFS)]},
            {"role": "assistant", "content": text},
        ],
    })


# ---------------------------------------------------------------- Essays
ESSAYS = {
    "Arbeidskrav 2 Kristendommen.md":
        "Skriv et arbeidskrav om et sentralt tema i kristendommen.",
    "Arbeidskrav- Buddhisme Hva er Nirvana og hvordan oppnår man det.md":
        "Skriv et arbeidskrav om hva nirvana er og hvordan man oppnår det i buddhismen.",
    "Semesteroppgave Kristendom & Jødedom - Nøkkelsymboler i Jonas Bok.md":
        "Skriv en semesteroppgave om nøkkelsymboler i Jonas' bok, i lys av kristendom og jødedom.",
    "Semesteroppgave høst HIB - Arvind .md":
        "Skriv en semesteroppgave i religionsvitenskap.",
}


def strip_md(text):
    text = re.sub(r"^#{1,6}\s*", "", text, flags=re.M)   # headings -> plain
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
    text = re.sub(r"\*(.+?)\*", r"\1", text)
    return text.strip()


def chunk_paragraphs(text, max_chars=5000):
    paras = re.split(r"\n\s*\n", text)
    chunks, cur = [], ""
    for p in paras:
        if len(cur) + len(p) > max_chars and cur:
            chunks.append(cur.strip()); cur = ""
        cur += p + "\n\n"
    if cur.strip():
        chunks.append(cur.strip())
    return chunks


def parse_essays(downloads):
    out = []
    for i, (fname, brief) in enumerate(ESSAYS.items()):
        p = Path(downloads) / fname
        if not p.exists():
            print(f"  ! essay not found, skipping: {fname}")
            continue
        body = strip_md(p.read_text(encoding="utf-8"))
        chunks = chunk_paragraphs(body)
        for j, ch in enumerate(chunks):
            out.append({
                "id": f"no-academic-essay-{i:02d}-{j:02d}",
                "date": "", "lang": "no", "register": "academic",
                "tier": "gold", "complete": True, "source": "essay-md",
                "context": "" if j == 0 else "(fortsettelse av samme oppgave)",
                "brief": brief if j == 0 else brief + " (fortsett teksten)",
                "text": ch,
            })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--downloads", default=str(Path.home() / "Downloads"))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    dl = Path(args.downloads)

    claude_json = next(dl.glob("data-*/conversations.json"), None) \
        or next(dl.glob("**/conversations.json"), None)
    discord_root = dl / "package" / "Messages"

    claude = parse_claude(claude_json) if claude_json else []
    discord = parse_discord(discord_root) if discord_root.exists() else []
    essays = parse_essays(dl)

    print(f"Claude  (his replies): {len(claude):5d} samples"
          f"  [{sum(len(s['messages'][2]['content']) for s in claude)//6} words]")
    print(f"Discord (his msgs):    {len(discord):5d} samples"
          f"  [{sum(len(s['messages'][2]['content']) for s in discord)//6} words]")
    print(f"Essays  (gold chunks): {len(essays):5d} samples"
          f"  [{sum(len(e['text']) for e in essays)//6} words]")

    if args.dry_run:
        print("\n--dry-run: nothing written. Samples preview:")
        for s in (claude[:1] + discord[:1]):
            print(json.dumps(s, ensure_ascii=False)[:300])
        return

    (REPO / "data" / "chat" / "claude.jsonl").write_text(
        "\n".join(json.dumps(s, ensure_ascii=False) for s in claude) + "\n",
        encoding="utf-8")
    (REPO / "data" / "chat" / "discord.jsonl").write_text(
        "\n".join(json.dumps(s, ensure_ascii=False) for s in discord) + "\n",
        encoding="utf-8")
    # append essays to corpus.jsonl (gold tier -> upsampled with the rest)
    with open(REPO / "data" / "corpus.jsonl", "a", encoding="utf-8") as f:
        for e in essays:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")

    print("\nwrote data/chat/claude.jsonl, data/chat/discord.jsonl, "
          "and appended essays to data/corpus.jsonl")


if __name__ == "__main__":
    main()
