# """
# Combine the three per-VLM result CSVs into:
#   1. An ensemble CSV: per (image_id, model) row, with each VLM's scores as
#      separate columns plus the ensemble mean per dimension and overall.
#   2. A generation-model summary CSV: mean/std of the ensemble scores,
#      grouped by generation model (sd, sdxl, galip, omnigen, omnigen2_*),
#      across all 50 images.

# Usage:
#   python aggregate_ensemble.py \
#       --qwen3vl results_qwen3vl.csv \
#       --internvl results_internvl.csv \
#       --minicpm results_minicpm.csv \
#       --out-detail ensemble_detail.csv \
#       --out-summary ensemble_summary.csv
# """

# import argparse
# import csv
# import statistics as st
# from collections import defaultdict

# DIMENSIONS = ["scientific_accuracy", "visual_clarity", "complexity_alignment"]


# def load_vlm_csv(path: str, vlm_name: str) -> dict:
#     """Returns {(image_id, model): {dim: value, ..., overall_score: v, reasoning: str}}"""
#     out = {}
#     with open(path, newline="", encoding="utf-8") as f:
#         for row in csv.DictReader(f):
#             key = (row["image_id"], row["model"])
#             out[key] = {
#                 "scientific_accuracy": float(row["scientific_accuracy"]),
#                 "visual_clarity": float(row["visual_clarity"]),
#                 "complexity_alignment": float(row["complexity_alignment"]),
#                 "overall_score": float(row["overall_score"]),
#                 "reasoning": row["reasoning"],
#             }
#     return out


# def main():
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--qwen3vl", required=True)
#     ap.add_argument("--internvl", required=True)
#     ap.add_argument("--minicpm", required=True)
#     ap.add_argument("--out-detail", default="ensemble_detail.csv")
#     ap.add_argument("--out-summary", default="ensemble_summary.csv")
#     args = ap.parse_args()

#     vlm_data = {
#         "qwen3vl": load_vlm_csv(args.qwen3vl, "qwen3vl"),
#         "internvl": load_vlm_csv(args.internvl, "internvl"),
#         "minicpm": load_vlm_csv(args.minicpm, "minicpm"),
#     }

#     all_keys = set()
#     for d in vlm_data.values():
#         all_keys.update(d.keys())

#     missing_report = []
#     detail_rows = []
#     for key in sorted(all_keys):
#         image_id, model = key
#         present = {name: d.get(key) for name, d in vlm_data.items()}
#         missing = [name for name, v in present.items() if v is None]
#         if missing:
#             missing_report.append((key, missing))

#         row = {"image_id": image_id, "model": model}
#         for vlm_name, v in present.items():
#             for dim in DIMENSIONS + ["overall_score"]:
#                 row[f"{vlm_name}_{dim}"] = v[dim] if v else ""

#         # Ensemble mean per dimension, computed only over VLMs that produced a score
#         for dim in DIMENSIONS:
#             vals = [v[dim] for v in present.values() if v]
#             row[f"ensemble_{dim}"] = round(sum(vals) / len(vals), 3) if vals else ""

#         overall_vals = [v["overall_score"] for v in present.values() if v]
#         row["ensemble_overall_score"] = round(sum(overall_vals) / len(overall_vals), 3) if overall_vals else ""
#         row["n_vlms_scored"] = len(overall_vals)

#         detail_rows.append(row)

#     detail_fields = (
#         ["image_id", "model"]
#         + [f"{name}_{dim}" for name in vlm_data for dim in DIMENSIONS + ["overall_score"]]
#         + [f"ensemble_{dim}" for dim in DIMENSIONS]
#         + ["ensemble_overall_score", "n_vlms_scored"]
#     )
#     with open(args.out_detail, "w", newline="", encoding="utf-8") as f:
#         writer = csv.DictWriter(f, fieldnames=detail_fields)
#         writer.writeheader()
#         writer.writerows(detail_rows)
#     print(f"Wrote {len(detail_rows)} rows to {args.out_detail}")

#     if missing_report:
#         print(f"WARNING: {len(missing_report)} (image_id, model) pairs are missing a score from at least one VLM:")
#         for key, missing in missing_report[:20]:
#             print(f"  {key}: missing {missing}")
#         if len(missing_report) > 20:
#             print(f"  ... and {len(missing_report) - 20} more")

#     # --- Generation-model-wise summary (mean/std across all images) ---
#     by_model = defaultdict(lambda: defaultdict(list))
#     for row in detail_rows:
#         if row["ensemble_overall_score"] == "":
#             continue
#         m = row["model"]
#         for dim in DIMENSIONS + ["overall_score"]:
#             colname = f"ensemble_{dim}" if dim != "overall_score" else "ensemble_overall_score"
#             by_model[m][dim].append(float(row[colname]))

#     summary_rows = []
#     for model, dims in sorted(by_model.items()):
#         row = {"model": model, "n_images": len(dims["overall_score"])}
#         for dim in DIMENSIONS + ["overall_score"]:
#             vals = dims[dim]
#             row[f"{dim}_mean"] = round(st.mean(vals), 3)
#             row[f"{dim}_std"] = round(st.pstdev(vals), 3) if len(vals) > 1 else 0.0
#         summary_rows.append(row)

#     summary_fields = (
#         ["model", "n_images"]
#         + [f"{dim}_{stat}" for dim in DIMENSIONS + ["overall_score"] for stat in ("mean", "std")]
#     )
#     with open(args.out_summary, "w", newline="", encoding="utf-8") as f:
#         writer = csv.DictWriter(f, fieldnames=summary_fields)
#         writer.writeheader()
#         writer.writerows(summary_rows)
#     print(f"Wrote {len(summary_rows)} model summaries to {args.out_summary}")


# if __name__ == "__main__":
#     main()

"""
Combine the per-VLM result CSVs (one per VLM in the ensemble) into:
  1. An ensemble CSV: per (image_id, model) row, with each VLM's scores as
     separate columns plus the ensemble mean per dimension and overall.
  2. A generation-model summary CSV: mean/std of the ensemble scores,
     grouped by generation model (stable_diffusion, sdxl, galip, omnigen,
     omnigen2), across all 50 images.

Usage (pass one --vlm NAME=path.csv per ensemble member):
  python aggregate_ensemble.py \
      --vlm qwen3vl=results_qwen3vl.csv \
      --vlm internvl=results_internvl.csv \
      --vlm deepseekvl=results_deepseekvl.csv \
      --out-detail ensemble_detail.csv \
      --out-summary ensemble_summary.csv
"""

import argparse
import csv
import statistics as st
from collections import defaultdict

DIMENSIONS = ["scientific_accuracy", "visual_clarity", "complexity_alignment"]


def load_vlm_csv(path: str) -> dict:
    """Returns {(image_id, model): {dim: value, ..., overall_score: v, reasoning: str}}"""
    out = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            key = (row["image_id"], row["model"])
            out[key] = {
                "scientific_accuracy": float(row["scientific_accuracy"]),
                "visual_clarity": float(row["visual_clarity"]),
                "complexity_alignment": float(row["complexity_alignment"]),
                "overall_score": float(row["overall_score"]),
                "reasoning": row["reasoning"],
            }
    return out


def parse_vlm_arg(spec: str):
    if "=" not in spec:
        raise argparse.ArgumentTypeError(
            f"--vlm expects NAME=path.csv, got: {spec!r}"
        )
    name, path = spec.split("=", 1)
    return name, path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--vlm", action="append", required=True, type=parse_vlm_arg, dest="vlms",
        metavar="NAME=path.csv",
        help="Repeatable. One per VLM in the ensemble, e.g. --vlm qwen3vl=results_qwen3vl.csv",
    )
    ap.add_argument("--out-detail", default="ensemble_detail.csv")
    ap.add_argument("--out-summary", default="ensemble_summary.csv")
    args = ap.parse_args()

    if len(args.vlms) < 2:
        raise SystemExit("Provide at least 2 --vlm entries to form an ensemble.")

    vlm_data = {name: load_vlm_csv(path) for name, path in args.vlms}

    all_keys = set()
    for d in vlm_data.values():
        all_keys.update(d.keys())

    missing_report = []
    detail_rows = []
    for key in sorted(all_keys):
        image_id, model = key
        present = {name: d.get(key) for name, d in vlm_data.items()}
        missing = [name for name, v in present.items() if v is None]
        if missing:
            missing_report.append((key, missing))

        row = {"image_id": image_id, "model": model}
        for vlm_name, v in present.items():
            for dim in DIMENSIONS + ["overall_score"]:
                row[f"{vlm_name}_{dim}"] = v[dim] if v else ""

        # Ensemble mean per dimension, computed only over VLMs that produced a score
        for dim in DIMENSIONS:
            vals = [v[dim] for v in present.values() if v]
            row[f"ensemble_{dim}"] = round(sum(vals) / len(vals), 3) if vals else ""

        overall_vals = [v["overall_score"] for v in present.values() if v]
        row["ensemble_overall_score"] = round(sum(overall_vals) / len(overall_vals), 3) if overall_vals else ""
        row["n_vlms_scored"] = len(overall_vals)

        detail_rows.append(row)

    vlm_names = list(vlm_data.keys())
    detail_fields = (
        ["image_id", "model"]
        + [f"{name}_{dim}" for name in vlm_names for dim in DIMENSIONS + ["overall_score"]]
        + [f"ensemble_{dim}" for dim in DIMENSIONS]
        + ["ensemble_overall_score", "n_vlms_scored"]
    )
    with open(args.out_detail, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=detail_fields)
        writer.writeheader()
        writer.writerows(detail_rows)
    print(f"Wrote {len(detail_rows)} rows to {args.out_detail}")

    if missing_report:
        print(f"WARNING: {len(missing_report)} (image_id, model) pairs are missing a score from at least one VLM:")
        for key, missing in missing_report[:20]:
            print(f"  {key}: missing {missing}")
        if len(missing_report) > 20:
            print(f"  ... and {len(missing_report) - 20} more")

    # --- Generation-model-wise summary (mean/std across all images) ---
    by_model = defaultdict(lambda: defaultdict(list))
    for row in detail_rows:
        if row["ensemble_overall_score"] == "":
            continue
        m = row["model"]
        for dim in DIMENSIONS + ["overall_score"]:
            colname = f"ensemble_{dim}" if dim != "overall_score" else "ensemble_overall_score"
            by_model[m][dim].append(float(row[colname]))

    summary_rows = []
    for model, dims in sorted(by_model.items()):
        row = {"model": model, "n_images": len(dims["overall_score"])}
        for dim in DIMENSIONS + ["overall_score"]:
            vals = dims[dim]
            row[f"{dim}_mean"] = round(st.mean(vals), 3)
            row[f"{dim}_std"] = round(st.pstdev(vals), 3) if len(vals) > 1 else 0.0
        summary_rows.append(row)

    summary_fields = (
        ["model", "n_images"]
        + [f"{dim}_{stat}" for dim in DIMENSIONS + ["overall_score"] for stat in ("mean", "std")]
    )
    with open(args.out_summary, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=summary_fields)
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"Wrote {len(summary_rows)} model summaries to {args.out_summary}")


if __name__ == "__main__":
    main()

