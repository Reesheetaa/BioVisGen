"""
Report which (image_id, generation_model) evaluations are still missing for
each VLM's results CSV, compared against the full expected task grid
(all image_ids x all config.GEN_MODELS).

Usage:
  python check_missing.py \
      --vlm qwen3vl=results_qwen3vl.csv \
      --vlm internvl=results_internvl.csv \
      --vlm deepseekvl=results_deepseekvl.csv \
      --vlm gemma4=results_gemma4.csv

Writes missing_<vlm_name>.csv (image_id, model) for each VLM that has gaps,
so you can see exactly what's left before re-running run_evaluation.py
(which will pick these up automatically thanks to its resume logic).
"""

import argparse
import csv
import os

import config
from data_utils import build_task_list, load_descriptions


def parse_vlm_arg(spec: str):
    if "=" not in spec:
        raise argparse.ArgumentTypeError(f"--vlm expects NAME=path.csv, got: {spec!r}")
    name, path = spec.split("=", 1)
    return name, path


def load_completed_keys(path: str) -> set:
    if not os.path.exists(path):
        return set()
    keys = set()
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            keys.add((row["image_id"], row["model"]))
    return keys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--vlm", action="append", required=True, type=parse_vlm_arg, dest="vlms",
        metavar="NAME=path.csv",
    )
    ap.add_argument("--images-root", default=config.IMAGES_ROOT)
    ap.add_argument("--descriptions-csv", default=config.DESCRIPTIONS_CSV)
    args = ap.parse_args()

    descriptions = load_descriptions(args.descriptions_csv)
    tasks = build_task_list(args.images_root, descriptions, config.GEN_MODELS,
                             config.REFERENCE_SUBFOLDER)
    expected_keys = {(t["image_id"], t["model"]) for t in tasks}
    print(f"Expected total tasks per VLM: {len(expected_keys)}\n")

    print(f"{'VLM':<15} {'done':>6} {'missing':>8}")
    print("-" * 32)

    any_missing = False
    for vlm_name, path in args.vlms:
        completed = load_completed_keys(path)
        missing = sorted(expected_keys - completed)
        extra = sorted(completed - expected_keys)  # rows for tasks no longer in the expected grid

        print(f"{vlm_name:<15} {len(completed):>6} {len(missing):>8}")

        if extra:
            print(f"  NOTE: {len(extra)} row(s) in {path} don't match the current expected "
                  f"task grid (e.g. from an old config/folder layout). Not counted as 'done'.")

        if missing:
            any_missing = True
            out_path = f"missing_{vlm_name}.csv"
            with open(out_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["image_id", "model"])
                writer.writerows(missing)
            print(f"  -> wrote {out_path}")

            # Break down missing count by generation model, useful to spot a
            # pattern (e.g. one gen-model failing systematically for this VLM)
            by_gen_model = {}
            for _image_id, gen_model in missing:
                by_gen_model[gen_model] = by_gen_model.get(gen_model, 0) + 1
            for gen_model, count in sorted(by_gen_model.items()):
                print(f"     {gen_model}: {count} missing")

    print()
    if any_missing:
        print("Re-run run_evaluation.py for any VLM with missing rows above — "
              "it will resume and only process what's missing.")
    else:
        print("All VLMs are complete for the current task grid. Safe to aggregate.")


if __name__ == "__main__":
    main()