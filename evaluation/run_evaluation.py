"""
Run ONE VLM over the full task grid (50 images x 7 gen-model variants) and
write results to its own CSV. Run this script once per VLM — since you have
two GPUs, you can run two of these in parallel (e.g. --gpu 0 for qwen3vl,
--gpu 1 for internvl, then --gpu 0 again for minicpm after qwen3vl finishes;
your two 48GB cards each comfortably fit one 8B VLM at a time, and could fit
two if you want to double them up on one card instead).

Usage:
  python run_evaluation.py --model qwen3vl --gpu 0 --output results_qwen3vl.csv
  python run_evaluation.py --model internvl --gpu 1 --output results_internvl.csv
  python run_evaluation.py --model minicpm --gpu 0 --output results_minicpm.csv

Resumable: if --output already exists, already-completed (image_id, model)
rows are skipped, so you can safely re-run after an interruption.
"""

import argparse
import csv
import importlib
import os
import sys
import time

import config
from data_utils import build_task_list, load_descriptions
from prompts import ResponseParseError, parse_vlm_json

CSV_FIELDS = [
    "image_id", "model", "scientific_accuracy", "visual_clarity",
    "complexity_alignment", "overall_score", "reasoning", "vlm",
]


def load_evaluator(vlm_key: str, gpu: int):
    entry = config.VLM_REGISTRY[vlm_key]
    module = importlib.import_module(entry["module"])
    cls = getattr(module, entry["class"])
    device = f"cuda:{gpu}"
    print(f"[run_evaluation] Loading {entry['hf_id']} on {device} ...")
    return cls(entry["hf_id"], device=device)


def already_done(output_csv: str):
    done = set()
    if os.path.exists(output_csv):
        with open(output_csv, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                done.add((row["image_id"], row["model"]))
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(config.VLM_REGISTRY.keys()))
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--images-root", default=config.IMAGES_ROOT)
    ap.add_argument("--descriptions-csv", default=config.DESCRIPTIONS_CSV)
    ap.add_argument("--output", required=True, help="Output CSV path for this VLM")
    ap.add_argument("--errors-log", default=None, help="Path to log unparseable responses (default: <output>.errors.log)")
    args = ap.parse_args()

    errors_log = args.errors_log or (args.output + ".errors.log")

    descriptions = load_descriptions(args.descriptions_csv)
    tasks = build_task_list(args.images_root, descriptions, config.GEN_MODELS,
                             config.REFERENCE_SUBFOLDER)
    print(f"[run_evaluation] {len(tasks)} total (image_id, model) tasks resolved.")

    done = already_done(args.output)
    if done:
        print(f"[run_evaluation] Resuming: {len(done)} tasks already completed, will skip.")

    evaluator = load_evaluator(args.model, args.gpu)

    write_header = not os.path.exists(args.output)
    out_f = open(args.output, "a", newline="", encoding="utf-8")
    writer = csv.DictWriter(out_f, fieldnames=CSV_FIELDS)
    if write_header:
        writer.writeheader()
        out_f.flush()

    n_ok, n_fail = 0, 0
    t0 = time.time()
    for i, task in enumerate(tasks):
        key = (task["image_id"], task["model"])
        if key in done:
            continue

        parsed = None
        last_raw = None
        for attempt in range(config.MAX_RETRIES):
            try:
                raw = evaluator.evaluate(
                    task["description"], task["ref_image_path"], task["gen_image_path"],
                    max_new_tokens=config.MAX_NEW_TOKENS, temperature=config.TEMPERATURE,
                )
                last_raw = raw
                parsed = parse_vlm_json(raw)
                break
            except ResponseParseError:
                continue
            except Exception as e:
                print(f"  [warn] generation error on {key}, attempt {attempt+1}: {e}", file=sys.stderr)
                continue

        if parsed is None:
            n_fail += 1
            with open(errors_log, "a", encoding="utf-8") as ef:
                ef.write(f"{key}\n{last_raw}\n{'-'*40}\n")
            print(f"[{i+1}/{len(tasks)}] FAILED to parse: {key}")
            continue

        row = {
            "image_id": task["image_id"],
            "model": task["model"],
            "scientific_accuracy": parsed["scientific_accuracy"],
            "visual_clarity": parsed["visual_clarity"],
            "complexity_alignment": parsed["complexity_alignment"],
            "overall_score": parsed["overall_score"],
            "reasoning": parsed["reasoning"],
            "vlm": args.model,
        }
        writer.writerow(row)
        out_f.flush()
        n_ok += 1

        if (i + 1) % 10 == 0:
            elapsed = time.time() - t0
            print(f"[{i+1}/{len(tasks)}] ok={n_ok} fail={n_fail} elapsed={elapsed/60:.1f}min")

    out_f.close()
    print(f"[run_evaluation] Done. ok={n_ok} fail={n_fail}. Output: {args.output}")
    if n_fail:
        print(f"[run_evaluation] See {errors_log} for unparseable responses (rerun script to retry them).")


if __name__ == "__main__":
    main()
