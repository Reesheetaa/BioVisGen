"""
Alternate aggregation view: one column per VLM ("critic") holding that VLM's
overall_score, plus row-wise average / std dev / median across the critics
for that (image_id, model) sample.

This is separate from aggregate_ensemble.py (which stays as the per-dimension
detail + generation-model summary view) -- use whichever output shape fits
your downstream analysis.

Usage:
  python aggregate_critics.py \
      --vlm qwen3vl=results_qwen3vl.csv \
      --vlm internvl=results_internvl.csv \
      --vlm deepseekvl=results_deepseekvl.csv \
      --vlm gemma4=results_gemma4.csv \
      --out critics_summary.csv

Output columns:
  image_id, model, <vlm1>_critic, <vlm2>_critic, ..., average, std_dev, median, n_critics_scored

- Each "<vlm>_critic" column is that VLM's overall_score (mean of its three
  dimension scores) for that (image_id, model) sample -- not the per-dimension
  breakdown.
- average/std_dev/median are computed ROW-WISE, across whichever critic
  columns have a score for that row (missing VLM scores for a row are
  skipped, not treated as zero). std_dev is population std (matches
  aggregate_ensemble.py's convention).
- n_critics_scored tells you how many of the VLMs actually contributed a
  score to that row's stats, so you can spot rows resting on fewer than the
  full ensemble (e.g. if a VLM has gaps -- run check_missing.py first).
"""

import argparse
import csv
import statistics as st


def parse_vlm_arg(spec: str):
    if "=" not in spec:
        raise argparse.ArgumentTypeError(f"--vlm expects NAME=path.csv, got: {spec!r}")
    name, path = spec.split("=", 1)
    return name, path


def load_overall_scores(path: str) -> dict:
    """Returns {(image_id, model): overall_score}."""
    out = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            key = (row["image_id"], row["model"])
            out[key] = float(row["overall_score"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--vlm", action="append", required=True, type=parse_vlm_arg, dest="vlms",
        metavar="NAME=path.csv",
    )
    ap.add_argument("--out", default="critics_summary.csv")
    args = ap.parse_args()

    if len(args.vlms) < 2:
        raise SystemExit("Provide at least 2 --vlm entries.")

    vlm_names = [name for name, _ in args.vlms]
    vlm_scores = {name: load_overall_scores(path) for name, path in args.vlms}

    all_keys = set()
    for scores in vlm_scores.values():
        all_keys.update(scores.keys())

    rows = []
    for key in sorted(all_keys):
        image_id, model = key
        row = {"image_id": image_id, "model": model}

        critic_vals = []
        for name in vlm_names:
            val = vlm_scores[name].get(key)
            row[f"{name}_critic"] = val if val is not None else ""
            if val is not None:
                critic_vals.append(val)

        if critic_vals:
            row["average"] = round(st.mean(critic_vals), 3)
            row["std_dev"] = round(st.pstdev(critic_vals), 3) if len(critic_vals) > 1 else 0.0
            row["median"] = round(st.median(critic_vals), 3)
        else:
            row["average"] = row["std_dev"] = row["median"] = ""
        row["n_critics_scored"] = len(critic_vals)

        rows.append(row)

    fieldnames = (
        ["image_id", "model"]
        + [f"{name}_critic" for name in vlm_names]
        + ["average", "std_dev", "median", "n_critics_scored"]
    )
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {args.out}")

    incomplete = [r for r in rows if r["n_critics_scored"] < len(vlm_names)]
    if incomplete:
        print(f"NOTE: {len(incomplete)} row(s) have fewer than {len(vlm_names)} critic "
              f"scores (missing VLM output for that sample). Run check_missing.py to see gaps.")


if __name__ == "__main__":
    main()