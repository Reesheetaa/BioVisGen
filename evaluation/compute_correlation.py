"""
OPTIONAL sanity-check script.

You said the goal is an alternative to human evaluation (not something that
needs to correlate with individual experts), so this isn't required for your
pipeline to be "done." But if you want a quick sanity check of how the AI
ensemble compares to your existing (aggregate) human scores, point this at
your human-scores CSV and it'll report Pearson + Spearman correlation,
overall and per-dimension.

Expects human_scores.csv with at least: image_id, model, and score column(s).
Edit HUMAN_COLUMNS below to match your actual column names.

Usage:
  python compute_correlation.py --ensemble ensemble_detail.csv --human human_scores.csv
"""

import argparse
import csv

from scipy.stats import pearsonr, spearmanr

# Map: our dimension name -> column name in YOUR human_scores.csv
# EDIT THESE to match your actual CSV headers before running.
HUMAN_COLUMNS = {
    "scientific_accuracy": "human_scientific_accuracy",
    "visual_clarity": "human_visual_clarity",
    "complexity_alignment": "human_complexity_alignment",
    "overall_score": "human_overall_score",
}


def load_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ensemble", default="ensemble_detail.csv")
    ap.add_argument("--human", required=True)
    args = ap.parse_args()

    ai_rows = {(r["image_id"], r["model"]): r for r in load_csv(args.ensemble)}
    human_rows = load_csv(args.human)

    missing_cols = [c for c in HUMAN_COLUMNS.values() if c not in human_rows[0]]
    if missing_cols:
        raise ValueError(
            f"human CSV is missing expected columns {missing_cols}. "
            f"Available columns: {list(human_rows[0].keys())}. "
            f"Edit HUMAN_COLUMNS in this script to match your CSV."
        )

    pairs = {dim: ([], []) for dim in HUMAN_COLUMNS}  # dim -> (ai_vals, human_vals)

    for hrow in human_rows:
        key = (hrow["image_id"], hrow["model"])
        arow = ai_rows.get(key)
        if arow is None or arow["ensemble_overall_score"] == "":
            continue
        for dim, human_col in HUMAN_COLUMNS.items():
            ai_col = "ensemble_overall_score" if dim == "overall_score" else f"ensemble_{dim}"
            try:
                ai_val = float(arow[ai_col])
                human_val = float(hrow[human_col])
            except (ValueError, KeyError):
                continue
            pairs[dim][0].append(ai_val)
            pairs[dim][1].append(human_val)

    print(f"{'dimension':<22} {'n':>4} {'pearson_r':>10} {'p':>8} {'spearman_r':>11} {'p':>8}")
    for dim, (ai_vals, human_vals) in pairs.items():
        n = len(ai_vals)
        if n < 3:
            print(f"{dim:<22} {n:>4}  (not enough paired points)")
            continue
        pr, pp = pearsonr(ai_vals, human_vals)
        sr, sp = spearmanr(ai_vals, human_vals)
        print(f"{dim:<22} {n:>4} {pr:>10.3f} {pp:>8.3f} {sr:>11.3f} {sp:>8.3f}")


if __name__ == "__main__":
    main()
