#!/usr/bin/env python3
"""Extract exam answers from Inspera candidate-report PDFs into corpus entries.

Inspera's PDF layout: metadata header, a task table, then per task the question
text followed by the marker "Skriv ditt svar her" (or "Write your answer here")
and the candidate's verbatim answer. School exams are written offline under
proctoring, which makes them verified-authorship (gold tier) — the whole point
of ingesting them.

Usage:
  python3 scripts/ingest_inspera.py --input folder_of_pdfs/ --out new_entries.jsonl
Requires PyMuPDF (pip install PyMuPDF).
"""

import argparse
import json
import re
from pathlib import Path

import fitz  # PyMuPDF

MARKERS = ["Skriv ditt svar her", "Write your answer here"]
# "2 EX-100 H24 Del 2" / "1 ST-205 V26 Oppgave" style section headers
SECTION_RE = re.compile(r"^\s*\d+\s+[A-ZÆØÅ]{2,4}[- ]?\d{2,3}\b.*$", re.M)

EN_HINTS = {"the", "and", "this", "that", "which", "government", "answer"}


def clean_pages(doc, course, title_lines):
    """Join page texts, dropping page furniture."""
    furniture = {course, "KANDIDAT", *title_lines}
    lines = []
    for page in doc:
        for ln in page.get_text().splitlines():
            s = ln.strip()
            if not s or s in furniture:
                continue
            if re.fullmatch(r"Candidate \d+|\d+/\d+|\d+", s):
                continue
            lines.append(ln)
    return "\n".join(lines)


def extract(pdf_path):
    doc = fitz.open(str(pdf_path))
    first = doc[0].get_text().splitlines()
    course = first[0].strip() if first else "UKJENT"
    date = ""
    for i, ln in enumerate(first):
        if ln.strip() == "Starttid" and i > 0:
            date = first[i - 1].strip().split(" ")[0]
    # page-header title lines repeat verbatim on every page; collect from page 2
    title_lines = set()
    if doc.page_count > 1:
        for ln in doc[1].get_text().splitlines()[:4]:
            if course.split("-")[0] in ln and "Candidate" not in ln:
                title_lines.add(ln.strip())
    text = clean_pages(doc, course, title_lines)

    entries = []
    marker_re = re.compile("|".join(re.escape(m) for m in MARKERS))
    positions = [m for m in marker_re.finditer(text)]
    for n, m in enumerate(positions):
        # answer: from marker to next section header (or next marker/end)
        after = text[m.end():]
        stops = [x.start() for x in (SECTION_RE.search(after),
                                     marker_re.search(after)) if x]
        answer = after[:min(stops)] if stops else after
        # drop the trailing publishing form / word counter Inspera appends
        tail_junk = re.search(
            r"Kan besvarelsen brukes|For å publisere|^\s*Publisering\s*$|\bOrd: ?\d+",
            answer, re.M)
        if tail_junk:
            answer = answer[:tail_junk.start()]
        # question: back to previous section header, capped to the text right
        # before the marker (front matter otherwise leaks into single-task exams)
        before = text[:m.start()]
        headers = list(SECTION_RE.finditer(before))
        q_start = headers[-1].end() if headers else 0
        question = before[q_start:].strip()[-1500:]
        for meta in ("PDF opprettet", "Sensurfrist", "Oppgavetype"):
            idx = question.rfind(meta)
            if idx != -1:
                question = question[idx + len(meta):].strip()
        q_open = re.search(r"(Oppgave|Spørsmål|Del \d|Besvar|Svar på|Case)", question)
        if q_open and q_open.start() > 0:
            question = question[q_open.start():]
        answer = answer.strip()
        if len(answer.split()) < 80:
            continue
        words = set(w.lower() for w in answer.split()[:200])
        lang = "en" if len(words & EN_HINTS) >= 3 else "no"
        entries.append({
            "id": f"exam-{course.lower().replace(' ', '')}-{n + 1}",
            "date": date or "unknown",
            "lang": lang,
            "register": "academic-exam",
            "tier": "gold",
            "complete": True,
            "source": f"inspera:{pdf_path.name}",
            "context": f"Skoleeksamen i {course} ({date}), skrevet i Inspera uten internett-tilgang",
            "brief": " ".join(question.split())[:1500],
            "text": answer,
        })
    return entries


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="folder of Inspera PDFs")
    ap.add_argument("--out", required=True, help="output JSONL (corpus-entry format)")
    args = ap.parse_args()

    all_entries = []
    for pdf in sorted(Path(args.input).glob("*.pdf")):
        found = extract(pdf)
        print(f"{pdf.name[:50]:52s} -> {len(found)} answer(s), "
              f"{sum(len(e['text'].split()) for e in found)} words")
        all_entries.extend(found)

    with open(args.out, "w", encoding="utf-8") as f:
        for e in all_entries:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    print(f"wrote {len(all_entries)} entries -> {args.out}")


if __name__ == "__main__":
    main()
