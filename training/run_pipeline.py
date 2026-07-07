#!/usr/bin/env python3
"""Two-stage training pipeline with live progress bars.

  Stage 1  SMOKE  — a handful of steps that exercises the whole path
                    (4-bit load, LoRA on the hybrid arch, packing, a save).
  gate      only continue if smoke exits 0 AND its final loss is finite.
  Stage 2  REAL   — the full 2-epoch QLoRA run.

Each stage streams train_qwythos.py's own progress bar straight to this
terminal, and both stages also drop a machine-readable status JSON
(training/status_smoke.json / status_real.json) that an external monitor
can poll. Run it and walk away:

  python training/run_pipeline.py                 # smoke -> real
  python training/run_pipeline.py --epochs 3      # real run length
  python training/run_pipeline.py --skip-smoke    # straight to the real run
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

# Children print a Unicode progress bar; keep their stdout UTF-8 on Windows.
CHILD_ENV = {**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"}

HERE = Path(__file__).resolve().parent
TRAIN = HERE / "train_qwythos.py"


def read_status(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return None


def run_stage(name, phase, epochs, out, status_file, env_note):
    """Launch one training stage, inheriting stdout so its live bar shows.
    Returns (exit_code, final_status_dict)."""
    status_file = Path(status_file)
    if status_file.exists():
        status_file.unlink()

    cmd = [sys.executable, str(TRAIN),
           "--epochs", str(epochs),
           "--out", str(out),
           "--status-file", str(status_file),
           "--phase", phase]

    bar = "=" * 62
    print(f"\n{bar}\n  STAGE: {name}   ({epochs} epochs -> {out})\n{bar}",
          flush=True)
    t0 = time.time()
    proc = subprocess.run(cmd, env=CHILD_ENV)
    dt = time.time() - t0

    status = read_status(status_file) or {}
    print(f"\n[{name}] exit={proc.returncode}  wall={dt/60:.1f} min", flush=True)
    return proc.returncode, status


def gate(status, max_step_seconds=45.0):
    """Decide whether the smoke stage looked healthy enough to proceed.
    Rejects on: error, non-finite/absent loss, or a per-step time so slow it
    means the fast kernels didn't engage (a full run would take many hours)."""
    if status.get("error"):
        return False, f"smoke reported error: {status['error']}"
    loss = status.get("loss")
    if loss is None:
        return False, "smoke produced no training loss"
    if loss != loss or loss in (float("inf"), float("-inf")):  # NaN/inf
        return False, f"smoke loss not finite ({loss})"
    if loss > 20:
        return False, f"smoke loss implausibly high ({loss})"
    rate = status.get("steps_per_s") or 0
    if rate > 0:
        sec_per_step = 1.0 / rate
        if sec_per_step > max_step_seconds:
            return False, (f"too slow: {sec_per_step:.0f}s/step — fast kernels "
                           f"likely inactive; a full run would take many hours")
        return True, (f"smoke healthy (loss {loss:.3f}, "
                      f"{sec_per_step:.1f}s/step)")
    return True, f"smoke healthy (final loss {loss:.3f})"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=float, default=2, help="real-run epochs")
    ap.add_argument("--smoke-epochs", type=float, default=0.03)
    ap.add_argument("--skip-smoke", action="store_true")
    ap.add_argument("--out", default=str(HERE / "lora_out_qwythos"))
    ap.add_argument("--smoke-out", default=str(HERE / "smoke_out"))
    args = ap.parse_args()

    if not args.skip_smoke:
        code, status = run_stage(
            "SMOKE", "smoke", args.smoke_epochs, args.smoke_out,
            HERE / "status_smoke.json", "")
        ok, why = gate(status) if code == 0 else (False, f"smoke exited {code}")
        print(f"\n>>> GATE: {'PASS' if ok else 'FAIL'} - {why}", flush=True)
        if not ok:
            print(">>> Aborting: not starting the real run.", flush=True)
            sys.exit(1)
        print(">>> Smoke passed. Auto-starting the real run.\n", flush=True)

    code, status = run_stage(
        "REAL", "train", args.epochs, args.out,
        HERE / "status_real.json", "")
    if code != 0 or status.get("error"):
        print(f"\n>>> REAL RUN FAILED (exit {code}): "
              f"{status.get('error','')}", flush=True)
        sys.exit(1)

    print(f"\n>>> DONE. Adapter saved to {args.out}/adapter", flush=True)


if __name__ == "__main__":
    main()
