#!/usr/bin/env python3
"""Merge the Arvind LoRA into the base Qwythos weights (fp16) so the result can be
converted to a single GGUF for llama.cpp / Ollama.

Qwythos is Qwen3_5ForConditionalGeneration (multimodal). GGUF text inference only
needs the language model, so after merging we try to save a text-only checkpoint:
the language_model submodule if present, else the whole thing (let the converter
pick the text part).

  python training/merge_lora.py --adapter training/lora_out_qwythos/adapter \
      --out /workspace/merged
"""

import argparse
import shutil
from pathlib import Path

import torch
from transformers import AutoModelForImageTextToText, AutoTokenizer
from peft import PeftModel

MODEL = "fluffy12222/Qwythos-9B-Claude-Mythos-5-1M-heretic-abliterated"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default="training/lora_out_qwythos/adapter")
    ap.add_argument("--out", default="/workspace/merged")
    args = ap.parse_args()
    out = Path(args.out)

    print("loading base in fp16 (no quant, needed for a clean merge) ...", flush=True)
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL, dtype=torch.float16, device_map="cpu", low_cpu_mem_usage=True)

    print(f"applying + merging adapter: {args.adapter}", flush=True)
    model = PeftModel.from_pretrained(model, args.adapter)
    model = model.merge_and_unload()

    # Save the FULL model. It keeps architecture "Qwen3_5ForConditionalGeneration",
    # which is exactly what llama.cpp's convert_hf_to_gguf.py is registered for
    # (conversion/qwen.py) — the converter strips the vision tower itself. An earlier
    # attempt to save only the text submodule wrote arch "Qwen3_5Model", which the
    # converter rejects ("Model Qwen3_5Model is not supported").
    model.save_pretrained(out, safe_serialization=True)
    print(f"saved FULL model (Qwen3_5ForConditionalGeneration) -> {out}", flush=True)

    # The adapter dir carries the patched chat template + tokenizer — copy those in
    # so the GGUF gets the same formatting training used.
    tok = AutoTokenizer.from_pretrained(args.adapter)
    tok.save_pretrained(out)
    for extra in ("chat_template.jinja",):
        src = Path(args.adapter) / extra
        if src.exists():
            shutil.copy(src, out / extra)
    print("tokenizer + chat template copied. done.", flush=True)


if __name__ == "__main__":
    main()
