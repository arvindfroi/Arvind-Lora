#!/usr/bin/env python3
"""Quick smell-test of the Arvind LoRA: base Qwythos + adapter, generate a few
Norwegian samples across registers. Not an eval — just "does it sound like him?".

  python training/chat_with_adapter.py
  python training/chat_with_adapter.py --adapter training/lora_out_qwythos/adapter
"""

import argparse
from pathlib import Path

import torch
from transformers import (AutoModelForImageTextToText, AutoTokenizer,
                          BitsAndBytesConfig)
from peft import PeftModel

MODEL = "fluffy12222/Qwythos-9B-Claude-Mythos-5-1M-heretic-abliterated"

# Matches prepare_training_data.py's SYSTEM_TEMPLATE exactly, so inference sees
# the same framing training did.
SYSTEM = (
    "Du er Arvind Frøiland. Du skriver nøyaktig slik Arvind skriver - med hans "
    "ordvalg, tegnsetting, skrivefeil og tone. Du polerer aldri teksten utover "
    "slik han faktisk skriver.\n"
    "Språk: {lang}. Register: {register}."
)

# (lang, register, user-brief) — a spread of the voices in the corpus.
PROMPTS = [
    ("norsk", "academic",
     "Skriv et kort avsnitt om hva nirvana er i buddhismen og hvordan man oppnår det."),
    ("norsk", "chat-ai",
     "Forklar kort hvorfor himmelen er blå."),
    ("norsk", "chat-snapchat",
     "(Svar en kompis som spør om du er med på LAN i helga.)"),
    ("engelsk", "chat-ai",
     "Give me 3 team ideas for a competitive Pokemon match."),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default="training/lora_out_qwythos/adapter")
    ap.add_argument("--max-new-tokens", type=int, default=220)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--base-only", action="store_true",
                    help="skip the adapter — shows what the raw model does, for contrast")
    args = ap.parse_args()

    adapter = Path(args.adapter)
    # The adapter dir carries the patched chat_template.jinja + tokenizer, so load
    # the tokenizer from there (falls back to base if running --base-only).
    tok_src = adapter if (adapter / "tokenizer_config.json").exists() else MODEL
    tokenizer = AutoTokenizer.from_pretrained(tok_src)

    print("loading base model in 4-bit ...", flush=True)
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16),
        dtype=torch.bfloat16, device_map={"": 0})

    if not args.base_only:
        print(f"applying adapter: {adapter}", flush=True)
        model = PeftModel.from_pretrained(model, str(adapter))
    model.config.use_cache = True
    model.eval()

    for lang, register, brief in PROMPTS:
        messages = [
            {"role": "system", "content": SYSTEM.format(lang=lang, register=register)},
            {"role": "user", "content": brief},
        ]
        inputs = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, enable_thinking=False,
            return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(
                inputs, max_new_tokens=args.max_new_tokens, do_sample=True,
                temperature=args.temperature, top_p=0.9,
                repetition_penalty=1.05,
                pad_token_id=tokenizer.eos_token_id)
        text = tokenizer.decode(out[0][inputs.shape[1]:], skip_special_tokens=True)
        print("\n" + "=" * 70)
        print(f"[{lang} / {register}]  {brief}")
        print("-" * 70)
        print(text.strip())
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
