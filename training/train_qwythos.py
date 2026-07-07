#!/usr/bin/env python3
"""QLoRA fine-tune on the Arvind corpus — Qwythos-9B base.

fluffy12222/Qwythos-9B-Claude-Mythos-5-1M-heretic-abliterated is a
Qwen3_5ForConditionalGeneration (hybrid linear-attention VLM, transformers>=5.13).
Unsloth doesn't know this architecture yet, so this uses plain
transformers + peft + trl. Text-only training; the vision tower stays frozen.

  python training/train_qwythos.py                 # real run, 2 epochs
  python training/train_qwythos.py --epochs 0.1    # smoke test
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

MODEL = "fluffy12222/Qwythos-9B-Claude-Mythos-5-1M-heretic-abliterated"
ROOT = Path(__file__).resolve().parent


def _fmt_eta(seconds):
    if seconds is None or seconds < 0:
        return "?"
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


def make_progress_callback(status_file, phase):
    """A TrainerCallback: writes an atomic status.json each step and paints a
    single-line progress bar to stdout. Kept import-light so the module still
    imports without transformers present.
    """
    from transformers import TrainerCallback

    status_path = Path(status_file) if status_file else None
    bar_width = 34

    class ProgressCallback(TrainerCallback):
        def __init__(self):
            self.start = None
            self.loss = None
            self.eval_loss = None
            self.last_t = None
            self.last_step = 0
            self.inst = None      # instantaneous seconds/step (excludes warmup)

        def _write(self, state, done=False, error=None):
            step = state.global_step
            total = state.max_steps or 0
            frac = (step / total) if total else 0.0
            elapsed = time.time() - self.start if self.start else 0.0
            # instantaneous rate from the last inter-step interval (ignores the
            # one-time Triton JIT/autotune cost baked into the first few steps)
            rate = (1.0 / self.inst) if self.inst else 0.0
            eta = (total - step) / rate if rate > 0 else None

            if status_path:
                payload = {
                    "phase": phase, "step": step, "total_steps": total,
                    "frac": round(frac, 4), "loss": self.loss,
                    "eval_loss": self.eval_loss,
                    "elapsed_s": round(elapsed, 1),
                    "steps_per_s": round(rate, 4),
                    "inst_s_per_step": round(self.inst, 2) if self.inst else None,
                    "eta_s": round(eta, 1) if eta else None,
                    "epoch": round(state.epoch, 3) if state.epoch else 0,
                    "done": done, "error": error, "ts": time.time(),
                }
                tmp = status_path.with_suffix(".tmp")
                tmp.write_text(json.dumps(payload), encoding="utf-8")
                os.replace(tmp, status_path)

            filled = int(bar_width * frac)
            bar = "█" * filled + "░" * (bar_width - filled)
            loss = f"{self.loss:.3f}" if self.loss is not None else "  -  "
            ev = f" ev {self.eval_loss:.3f}" if self.eval_loss is not None else ""
            sys.stdout.write(
                f"\r[{phase}] |{bar}| {frac*100:5.1f}%  "
                f"{step}/{total}  loss {loss}{ev}  "
                f"{rate:.2f} it/s  eta {_fmt_eta(eta)}   ")
            sys.stdout.flush()

        def on_train_begin(self, args, state, control, **kw):
            self.start = time.time()
            self._write(state)

        def on_log(self, args, state, control, logs=None, **kw):
            logs = logs or {}
            if "loss" in logs:
                self.loss = logs["loss"]
            if "eval_loss" in logs:
                self.eval_loss = logs["eval_loss"]

        def on_step_end(self, args, state, control, **kw):
            now = time.time()
            if self.last_t is not None and state.global_step > self.last_step:
                self.inst = (now - self.last_t) / (state.global_step - self.last_step)
            self.last_t, self.last_step = now, state.global_step
            self._write(state)

        def on_evaluate(self, args, state, control, metrics=None, **kw):
            if metrics and "eval_loss" in metrics:
                self.eval_loss = metrics["eval_loss"]
            self._write(state)

        def on_train_end(self, args, state, control, **kw):
            self._write(state, done=True)
            sys.stdout.write("\n")
            sys.stdout.flush()

    return ProgressCallback()


def lora_targets(model):
    """All attention/MLP linear projections in the text stack, by suffix.

    The hybrid arch mixes full-attention blocks (q/k/v/o_proj) with gated
    DeltaNet blocks (in_proj_*, out_proj etc.), so discover names instead of
    hardcoding. Vision tower and lm_head are excluded.
    """
    import torch.nn as nn
    import bitsandbytes as bnb

    suffixes = set()
    for name, module in model.named_modules():
        if not isinstance(module, (nn.Linear, bnb.nn.Linear4bit)):
            continue
        if "visual" in name or "vision" in name or "lm_head" in name:
            continue
        if "embed" in name or "mtp" in name:
            continue
        suffixes.add(name.split(".")[-1])
    return sorted(suffixes)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=float, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--seq-len", type=int, default=2048)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--packing", action="store_true",
                    help="pack samples into one sequence. ONLY safe with a "
                         "FlashAttention varlen backend; without it, samples "
                         "cross-contaminate. Off by default for this arch.")
    ap.add_argument("--attn", default=None,
                    help="attn_implementation, e.g. flash_attention_2 (cloud "
                         "GPUs). Required for correct+fast --packing.")
    ap.add_argument("--out", default=str(ROOT / "lora_out_qwythos"))
    ap.add_argument("--status-file", default=None,
                    help="write live progress JSON here for the launcher/monitor")
    ap.add_argument("--phase", default="train",
                    help="label shown in the progress bar")
    args = ap.parse_args()

    # Windows stdout defaults to cp1252 when redirected; the bar's block glyphs
    # (█ ░) would raise a charmap error. Force UTF-8, tolerate the rest.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    def write_status(**extra):
        if not args.status_file:
            return
        p = Path(args.status_file)
        payload = {"phase": args.phase, "step": 0, "total_steps": 0,
                   "frac": 0.0, "ts": time.time(), **extra}
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload), encoding="utf-8")
        os.replace(tmp, p)

    write_status(stage="loading_model")
    print(f"[{args.phase}] loading base model in 4-bit ...", flush=True)

    import torch
    from transformers import AutoModelForImageTextToText, AutoTokenizer, BitsAndBytesConfig
    from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
    from datasets import load_dataset
    from trl import SFTConfig, SFTTrainer

    tokenizer = AutoTokenizer.from_pretrained(MODEL)

    # The stock template hardcodes a "You are Qwythos ... never claim to be
    # anyone else" identity block after every system message, which fights the
    # Arvind persona head-on. Strip it; the patched template ships with the
    # adapter so inference sees the same format as training.
    import re
    tokenizer.chat_template = re.sub(
        r'set qwythos_identity = ".*?" %\}',
        'set qwythos_identity = "" %}',
        tokenizer.chat_template, count=1)

    # This is a reasoning model: at inference the template always opens the
    # assistant turn with a think block (empty one when enable_thinking=False),
    # but renders think-less training targets without it. Render assistant
    # targets with the empty think block so training matches no-think inference.
    plain = "{{- '<|im_start|>' + message.role + '\\n' + content }}"
    nothink = ("{{- '<|im_start|>' + message.role"
               " + '\\n<think>\\n\\n</think>\\n\\n' + content }}")
    assert tokenizer.chat_template.count(plain) == 1
    tokenizer.chat_template = tokenizer.chat_template.replace(plain, nothink)

    model = AutoModelForImageTextToText.from_pretrained(
        MODEL,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        ),
        dtype=torch.bfloat16,
        device_map={"": 0},
        attn_implementation=args.attn,
    )
    model.config.use_cache = False

    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    targets = lora_targets(model)
    print(f"LoRA target modules: {targets}")
    model = get_peft_model(model, LoraConfig(
        r=32,
        lora_alpha=64,
        lora_dropout=0.05,
        target_modules=targets,
        exclude_modules=r".*(visual|vision).*",
        task_type="CAUSAL_LM",
    ))
    model.print_trainable_parameters()

    data = load_dataset("json", data_files={
        "train": str(ROOT / "out" / "train.jsonl"),
        "val": str(ROOT / "out" / "val.jsonl"),
    })

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=data["train"],
        eval_dataset=data["val"],
        args=SFTConfig(
            max_length=args.seq_len,
            packing=args.packing,
            # With packing off, variable batch lengths make fla autotune a fresh
            # Triton kernel per sequence-length -> a compile storm. Rounding every
            # batch up to a multiple of 256 bounds it to ~8 distinct shapes, so
            # autotune happens a handful of times, then runs cached.
            pad_to_multiple_of=256,
            per_device_train_batch_size=args.batch,
            gradient_accumulation_steps=args.grad_accum,
            num_train_epochs=args.epochs,
            learning_rate=args.lr,
            lr_scheduler_type="cosine",
            warmup_ratio=0.05,
            logging_steps=20,
            eval_strategy="steps",
            eval_steps=200,
            per_device_eval_batch_size=1,
            save_strategy="epoch",
            output_dir=args.out,
            seed=42,
            bf16=True,
            gradient_checkpointing=True,
            report_to="none",
            disable_tqdm=True,
        ),
        callbacks=[make_progress_callback(args.status_file, args.phase)],
    )
    trainer.train()

    model.save_pretrained(f"{args.out}/adapter")
    tokenizer.save_pretrained(f"{args.out}/adapter")
    write_status(stage="saved", done=True, out=f"{args.out}/adapter")
    print(f"done -> {args.out}/adapter")


def _status_path_from_argv():
    argv = sys.argv
    for i, a in enumerate(argv):
        if a == "--status-file" and i + 1 < len(argv):
            return argv[i + 1]
        if a.startswith("--status-file="):
            return a.split("=", 1)[1]
    return None


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # surface the failure into the status file
        import traceback
        traceback.print_exc()
        path = _status_path_from_argv()
        if path:
            try:
                Path(path).write_text(json.dumps(
                    {"error": str(exc), "done": False, "ts": time.time()}),
                    encoding="utf-8")
            except Exception:
                pass
        sys.exit(1)
