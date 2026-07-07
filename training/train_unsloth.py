#!/usr/bin/env python3
"""QLoRA fine-tune on the Arvind corpus with Unsloth.

Run on any 16-24 GB GPU (RunPod/Vast RTX 4090, Colab A100) — see training/README.md.

  pip install unsloth
  python3 training/train_unsloth.py --model gemma          # the real run
  python3 training/train_unsloth.py --model llama --epochs 1   # cheap prototype
"""

import argparse
from pathlib import Path

MODELS = {
    "gemma": "unsloth/gemma-3-12b-it-bnb-4bit",
    "llama": "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit",
}

ROOT = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=MODELS, default="gemma")
    ap.add_argument("--epochs", type=float, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--seq-len", type=int, default=2048)
    ap.add_argument("--out", default=str(ROOT / "lora_out"))
    args = ap.parse_args()

    from unsloth import FastLanguageModel
    from datasets import load_dataset
    from trl import SFTConfig, SFTTrainer

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODELS[args.model],
        max_seq_length=args.seq_len,
        load_in_4bit=True,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=32,
        lora_alpha=64,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )

    data = load_dataset("json", data_files={
        "train": str(ROOT / "out" / "train.jsonl"),
        "val": str(ROOT / "out" / "val.jsonl"),
    })

    def to_text(row):
        return {"text": tokenizer.apply_chat_template(
            row["messages"], tokenize=False, add_generation_prompt=False)}

    data = data.map(to_text, remove_columns=data["train"].column_names)

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=data["train"],
        eval_dataset=data["val"],
        args=SFTConfig(
            dataset_text_field="text",
            max_seq_length=args.seq_len,
            packing=True,                      # short chat samples -> big speedup
            per_device_train_batch_size=2,
            gradient_accumulation_steps=8,     # effective batch 16
            num_train_epochs=args.epochs,
            learning_rate=args.lr,
            lr_scheduler_type="cosine",
            warmup_ratio=0.05,
            logging_steps=20,
            eval_strategy="steps",
            eval_steps=200,
            save_strategy="epoch",
            output_dir=args.out,
            seed=42,
            bf16=True,
            report_to="none",
        ),
    )
    trainer.train()

    model.save_pretrained(f"{args.out}/adapter")
    tokenizer.save_pretrained(f"{args.out}/adapter")
    # GGUF for Ollama (q4_k_m keeps quality, ~7 GB for 12B):
    model.save_pretrained_gguf(f"{args.out}/gguf", tokenizer,
                               quantization_method="q4_k_m")
    print(f"done -> {args.out}/adapter (LoRA) and {args.out}/gguf (Ollama-ready)")


if __name__ == "__main__":
    main()
