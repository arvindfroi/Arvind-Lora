# Getting your chat data out of each platform

None of these platforms allow direct API access to message history, so the pipeline
consumes their official data exports. Do these on your own devices, then drop the
files into this repo (or a private location) and run `scripts/ingest_chats.py`.

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
