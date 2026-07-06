# Getting your chat data out of each platform

None of these platforms allow direct API access to message history, so the pipeline
consumes their official data exports. Do these on your own devices, then drop the
files into this repo (or a private location) and run `scripts/ingest_chats.py`.

## The hand-off (no terminal needed on your side)

**Upload any export zip/file to a Google Drive folder named `lora-exports`** and say
so in the Claude session. Claude can read your Drive, so it will download the export,
run the ingestion, review/anonymize, and commit the training samples — you never have
to run the scripts yourself. (Status as of 2026-07-05: no Snapchat export, Google
Takeout, or Apple data request has ever been run on these accounts — each one below
still needs you to click through its export flow once, because they require your
password/2FA.)

## Snapchat (your main one)

1. Go to <https://accounts.snapchat.com> → **My Data**.
2. Select at minimum **Chat History**, choose **JSON** format, submit the request.
3. Snapchat emails you a download link (can take hours–days). Download and unzip.
4. Ingest:
   ```bash
   python3 scripts/ingest_chats.py --platform snapchat \
     --input mydata/json/chat_history.json --me "your_snap_username"
   ```

**Heads-up:** Snapchat deletes unsaved chats, so the export only contains saved
chats and a limited window. Turn on chat-saving with close friends going forward if
you want this stream to grow.

## iMessage (needs your Mac, 10 minutes)

1. On your Mac: `brew install imessage-exporter`
2. Give your terminal Full Disk Access (System Settings → Privacy & Security),
   then run:
   ```bash
   imessage-exporter -f txt -o ~/imessage-export
   ```
3. Ingest:
   ```bash
   python3 scripts/ingest_chats.py --platform imessage --input ~/imessage-export
   ```
   Your own messages are labeled `Me` in the export, which the script already
   recognizes.

## WhatsApp

Per chat (pick your 10–20 most active): chat → contact name → **Export Chat** →
**Without Media**. Then:
```bash
python3 scripts/ingest_chats.py --platform whatsapp --input _chat.txt --me "Arvind"
```

## Instagram DMs & Facebook Messenger

1. <https://accountscenter.meta.com> → Your information and permissions →
   **Download your information** → select Messages, format **JSON**.
2. Unzip; ingest the whole inbox at once:
   ```bash
   python3 scripts/ingest_chats.py --platform instagram \
     --input your_instagram_activity/messages/inbox/ --me "Arvind Frøiland"
   ```

## Android SMS/RCS (you're on a Pixel — this one is easy and rich)

Google Takeout does **not** include Messages content; the standard route is the
free **SMS Backup & Restore** app (SyncTech):

1. Install it from Play Store → Back up → Messages only, local backup, XML.
2. Upload the `sms-*.xml` file to Drive (`lora-exports`).
3. Ingest: `python3 scripts/ingest_chats.py --platform sms --input sms-20260705.xml`

## Generating NEW dialect data (live tool)

The dialect (sørvestlandsk) is underrepresented outside Snapchat, so there's a
purpose-built capture tool deployed on Val Town:

**https://arvindfroi--0142b74c797411f1b67e1607ee4eb77e.web.val.run**

- Open it on your phone, answer the prompt exactly like you'd talk, hit "Lagre og
  neste". 22 rotating prompts across chat/storytelling/opinion registers, most
  dialect-eliciting, a couple bokmål for contrast. 2–5 minutes a day.
- Every answer is stored as a ready gold-tier corpus entry (the prompt becomes the
  training brief). Claude pulls them with:
  `curl <tool-url>/export-x7qk2m9v.jsonl >> data/corpus.jsonl` (then dedupe/validate).
- Source code lives in the val `arvindfroi/dialekt-treningsdata`.

**Voice route (highest dialect density):** record yourself talking (voice memos,
rants, explaining things) and upload the audio files to Drive. Transcription with
NB-Whisper (Nasjonalbiblioteket's Norwegian Whisper, handles dialects) turns
speech into `spoken-dialect` register samples. Spoken ≠ written, so these get
their own register tag and moderate weight — but nothing captures dialect faster.

## Gboard (honest assessment: skip it)

Gboard does not upload the text you type — there is nothing to export that contains
your prose. The only exportable artifact is the personal dictionary (Gboard →
Dictionary → Personal dictionary → ⋮ → Export), which is a word list, not writing.
Useful only as a slang glossary to check the corpus against; not training data.
Your actual phone typing is better captured via SMS export (above) and the chat
platforms' own exports.

## Apple iCloud

Two separate things people mean by "iCloud data":

- **iMessages: NOT included in Apple's data export.** Apple's Data & Privacy
  download (<https://privacy.apple.com>) deliberately excludes message content.
  The only real route is the Mac `imessage-exporter` flow in the iMessage section
  above — 10 minutes on your Mac.
- **Notes, Pages documents, iCloud Drive files** (often full of your writing):
  - Quick per-item route: on the Mac, select notes → File → Export as PDF, or for
    bulk use the free [Exporter](https://apps.apple.com/app/exporter/id1099120373)
    app (Notes → Markdown files). Pages docs: File → Export To → Plain Text.
  - Bulk route: <https://privacy.apple.com> → request a copy of your data → select
    iCloud Drive files and Notes; Apple emails a download within ~7 days.
  - Upload whatever comes out to `lora-exports` in Drive; Markdown/txt/PDF all work —
    Claude will sort authored writing from clutter during curation.

## Discord

Discord's own data package doesn't include message text in a usable per-channel
form for third parties; the practical route is
[DiscordChatExporter](https://github.com/Tyrrrz/DiscordChatExporter) (JSON output)
for your DMs. A parser can be added to `ingest_chats.py` when you have a file —
open an issue/note in the repo.

## After ingesting

- Output lands in `data/chat/<platform>.jsonl`, already in training format.
- **Skim each file before training.** Delete anything you wouldn't want a model to
  memorize (other people's secrets, codes, addresses). Partners are already
  anonymized to `Partner1`, `Partner2`, … by the script.
- Concatenate with the email dataset for training:
  ```bash
  python3 scripts/build_dataset.py --tiers gold,silver --out out/
  cat data/chat/*.jsonl >> out/train.jsonl
  ```
  (Keep `out/val.jsonl` email-only, or hold out ~5% of chat lines too.)

## Privacy reminder

Chat exports contain **other people's** messages. The ingest script only puts your
own words in the assistant turns, but the context windows quote what partners wrote.
Keep this repo private, and consider trimming conversations with people who would
mind. The trained model will speak like you — it's on you where its words go.
