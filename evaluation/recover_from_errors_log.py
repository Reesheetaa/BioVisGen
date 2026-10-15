"""
Recover rows from a run_evaluation.py errors log using the (now-fixed)
robust JSON parser in prompts.py, WITHOUT re-running the VLM.

Each entry in <output>.errors.log looks like:
  ('image_id', 'model')
  <raw model text, possibly with trailing hallucinated content>
  ----------------------------------------

This re-parses each raw text with the current prompts.parse_vlm_json (which
correctly extracts the first balanced JSON object and ignores anything the
model generated after it -- fixing the earlier greedy-regex bug). Any entry
that parses successfully AND isn't already present in the target results
CSV gets appended. Anything still unparseable is left alone -- rerun
run_evaluation.py afterward to have the VLM retry those (resume logic will
only touch what's still missing).

Usage:
  python recover_from_errors_log.py \
      --errors-log results_deepseekvl.csv.errors.log \
      --output results_deepseekvl.csv \
      --vlm-name deepseekvl
"""

import argparse
import ast
import csv
import os
import re

from prompts import parse_vlm_json, ResponseParseError

CSV_FIELDS = [
    "image_id", "model", "scientific_accuracy", "visual_clarity",
    "complexity_alignment", "overall_score", "reasoning", "vlm",
]


def load_completed_keys(path: str) -> set:
    keys = set()
    if os.path.exists(path):
        with open(path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                keys.add((row["image_id"], row["model"]))
    return keys


def parse_log(path: str):
    """Split the errors log into (key, raw_text) entries."""
    text = open(path, encoding="utf-8").read()
    chunks = re.split(r"\n-{10,}\n?", text)
    entries = []
    for chunk in chunks:
        chunk = chunk.strip("\n")
        if not chunk.strip():
            continue
        first_nl = chunk.find("\n")
        if first_nl == -1:
            continue
        key_line = chunk[:first_nl].strip()
        raw_text = chunk[first_nl + 1:]
        try:
            key = ast.literal_eval(key_line)
        except (ValueError, SyntaxError):
            continue
        entries.append((key, raw_text))
    return entries


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--errors-log", required=True)
    ap.add_argument("--output", required=True, help="The results CSV to append recovered rows to")
    ap.add_argument("--vlm-name", required=True, help="Value to write in the 'vlm' column")
    args = ap.parse_args()

    completed = load_completed_keys(args.output)
    entries = parse_log(args.errors_log)
    print(f"Parsed {len(entries)} entries from {args.errors_log}")

    write_header = not os.path.exists(args.output)
    recovered, skipped_already_done, still_failed = 0, 0, 0
    rows_to_write = []
    seen_in_this_run = set()

    for key, raw_text in entries:
        if key in completed or key in seen_in_this_run:
            skipped_already_done += 1
            continue
        try:
            parsed = parse_vlm_json(raw_text)
        except ResponseParseError:
            still_failed += 1
            print(f"  still unparseable, needs real rerun: {key}")
            continue

        rows_to_write.append({
            "image_id": key[0],
            "model": key[1],
            "scientific_accuracy": parsed["scientific_accuracy"],
            "visual_clarity": parsed["visual_clarity"],
            "complexity_alignment": parsed["complexity_alignment"],
            "overall_score": parsed["overall_score"],
            "reasoning": parsed["reasoning"],
            "vlm": args.vlm_name,
        })
        seen_in_this_run.add(key)
        recovered += 1

    with open(args.output, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if write_header:
            writer.writeheader()
        for row in rows_to_write:
            writer.writerow(row)

    print(f"\nRecovered and appended: {recovered}")
    print(f"Already done, skipped (avoided duplicate): {skipped_already_done}")
    print(f"Still unparseable (rerun run_evaluation.py to retry these): {still_failed}")


if __name__ == "__main__":
    main()