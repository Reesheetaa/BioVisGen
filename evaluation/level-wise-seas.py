"""
Standalone script: QIG ensemble (Qwen3-VL + InternVL + Gemma4) scores for the
omnigen2_basic and omnigen2_advanced caption-complexity levels, across a
50-image list. (omnigen2 default level already calculated elsewhere, not
repeated here.)

Self-contained -- run directly, no dependency on other notebook cells.
Must be run from inside vlm_eval_pipeline/ (needs config.py, prompts.py,
models/).

Usage:
  python qig_levels_eval.py
"""

import os
import csv
import glob
import importlib
import statistics as st

import torch
import pandas as pd

import config
from prompts import parse_vlm_json, ResponseParseError

# ---------------------------------------------------------------------------
# Paste your full 50-image list here (21264 and 21936 included as examples)
# ---------------------------------------------------------------------------
# IMAGE_IDS_50 = ["21264", "21936"]  # <-- add the remaining 48 image_ids
IMAGE_IDS_50 =[
    "20067",
    "20356",
    "20677",
    "20732",
    "20845",
    "20851",
    "20978",
    "21260",
    "21264",
    "21316",
    "21380",
    "21886",
    "21936",
    "21963",
    "21991",
    "22024",
    "22486",
    "23339",
    "23360",
    "23500",
    "23627",
    "24053",
    "24273",
    "25498",
    "17180",
    "16262",
    "12006",
    "12653",
    "12779",
    "10084",
    "10120",
    "2077",
    "2091",
    "2122",
    "255",
    "4011",
    "4699",
    "4721",
    "4952",
    "513",
    "5610",
    "5944",
    "6437",
    "6485",
    "6581",
    "8269",
    "8960",
    "4668",
    "8999",
    "15218"
]


metadata_paths = {
    "omnigen2_basic":
        "data/metadata_beginner.csv",
    "omnigen2_advanced":
        "data/metadata_advanced.csv",
}

results_folders = {
    "omnigen2_basic":
        "."
        "omnigen2_17kbeginner_test_inference_1807/"
        "checkpoint-32320",
    "omnigen2_advanced":
        "."
        "omnigen2_17kadvanced_test_inference_2807/"
        "checkpoint-27472",
}

# omnigen2 (default) already calculated -- only doing basic + advanced here
LEVELS = {
    "omnigen2_basic": {
        "metadata_path": metadata_paths["omnigen2_basic"],
        "images_folder": results_folders["omnigen2_basic"],
    },
    "omnigen2_advanced": {
        "metadata_path": metadata_paths["omnigen2_advanced"],
        "images_folder": results_folders["omnigen2_advanced"],
    },
}

QIG_VLMS = ["qwen3vl", "internvl", "gemma4"]  # the QIG ensemble
GPU_DEVICE = "cuda:0"  # edit to whichever GPU is free

IMAGE_EXTS = ("png", "jpg", "jpeg", "webp", "bmp", "tif", "tiff")
CANDIDATE_CAPTION_COLUMNS = ["description", "caption", "text"]

OUT_CSV = "levelwise_qig_summary.csv"


def resolve_by_stem(folder, stem):
    matches = []
    for ext in IMAGE_EXTS:
        matches += glob.glob(os.path.join(folder, f"{stem}.{ext}"))
        matches += glob.glob(os.path.join(folder, f"{stem}.{ext.upper()}"))
    matches = sorted(set(matches))
    if not matches:
        raise FileNotFoundError(f"No image for stem {stem!r} in {folder}")
    if len(matches) > 1:
        raise FileNotFoundError(f"Multiple images for stem {stem!r} in {folder}: {matches}")
    return matches[0]


# def load_captions(csv_path):
#     """Returns {image_id: caption}. Auto-detects the caption column name."""
#     with open(csv_path, newline="", encoding="utf-8") as f:
#         reader = csv.DictReader(f)
#         fieldnames = reader.fieldnames
#         caption_col = next((c for c in CANDIDATE_CAPTION_COLUMNS if c in fieldnames), None)
#         if caption_col is None:
#             raise ValueError(
#                 f"Couldn't find a caption column in {csv_path}. "
#                 f"Tried {CANDIDATE_CAPTION_COLUMNS}, actual columns: {fieldnames}. "
#                 f"Add the real column name to CANDIDATE_CAPTION_COLUMNS above."
#             )
#         out = {}
#         for row in reader:
#             out[row["image"].strip()] = row[caption_col].strip()
#     return out

def load_captions(csv_path):
    """Returns {image_id: caption}. Extracts image id from full image path."""

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        fieldnames = reader.fieldnames
        print(f"Reading {csv_path}")
        print("Columns found:", fieldnames)

        caption_col = next(
            (c for c in CANDIDATE_CAPTION_COLUMNS if c in fieldnames),
            None
        )

        if caption_col is None:
            raise ValueError(
                f"Cannot find caption column. Available columns: {fieldnames}"
            )

        out = {}

        for row in reader:
            # Extract filename from full path
            image_path = row["image"].strip()

            # /path/to/image.png -> image.png
            filename = os.path.basename(image_path)

            # 21316.png -> 21316
            image_id = os.path.splitext(filename)[0]

            out[image_id] = row[caption_col].strip()

    return out


def build_tasks():
    captions_by_level = {level: load_captions(cfg["metadata_path"]) for level, cfg in LEVELS.items()}

    tasks = []
    missing = []
    for image_id in IMAGE_IDS_50:
        try:
            ref_path = resolve_by_stem(os.path.join(config.IMAGES_ROOT, config.REFERENCE_SUBFOLDER), image_id)
        except FileNotFoundError as e:
            missing.append(str(e))
            continue

        for level, cfg in LEVELS.items():
            caption = captions_by_level[level].get(image_id)
            if caption is None:
                missing.append(f"image_id {image_id} missing from {cfg['metadata_path']}")
                continue
            try:
                gen_path = resolve_by_stem(cfg["images_folder"], image_id)
            except FileNotFoundError as e:
                missing.append(str(e))
                continue

            tasks.append({
                "image_id": image_id,
                "model": level,
                "description": caption,
                "ref_image_path": ref_path,
                "gen_image_path": gen_path,
            })

    print(f"{len(tasks)} tasks resolved (expect {len(IMAGE_IDS_50)} images x {len(LEVELS)} levels "
          f"= {len(IMAGE_IDS_50) * len(LEVELS)})")
    if missing:
        print(f"\n{len(missing)} unresolved -- these will be skipped:")
        for m in missing[:20]:
            print(f"  - {m}")
        if len(missing) > 20:
            print(f"  ... and {len(missing) - 20} more")
    return tasks


def run_qig_inference(tasks):
    """Run each QIG critic sequentially (load -> score all tasks -> unload)."""
    all_scores = {}  # {(image_id, model): {vlm_name: overall_score_or_None}}

    for vlm_key in QIG_VLMS:
        entry = config.VLM_REGISTRY[vlm_key]
        module = importlib.import_module(entry["module"])
        cls = getattr(module, entry["class"])
        print(f"Loading {entry['hf_id']} on {GPU_DEVICE} ...")
        evaluator = cls(entry["hf_id"], device=GPU_DEVICE)

        for t in tasks:
            key = (t["image_id"], t["model"])
            parsed = None
            for attempt in range(config.MAX_RETRIES):
                try:
                    raw = evaluator.evaluate(
                        t["description"], t["ref_image_path"], t["gen_image_path"],
                        max_new_tokens=config.MAX_NEW_TOKENS, temperature=config.TEMPERATURE,
                    )
                    parsed = parse_vlm_json(raw)
                    break
                except ResponseParseError:
                    continue
                except Exception as e:
                    print(f"  [warn] {vlm_key} error on {key}, attempt {attempt + 1}: {e}")
                    continue
            all_scores.setdefault(key, {})[vlm_key] = parsed["overall_score"] if parsed else None
            if parsed is None:
                print(f"  FAILED to parse: {vlm_key} on {key}")

        del evaluator
        torch.cuda.empty_cache()
        print(f"Done with {vlm_key}\n")

    return all_scores


def aggregate(tasks, all_scores):
    rows = []
    for t in tasks:
        key = (t["image_id"], t["model"])
        scores = all_scores.get(key, {})
        row = {"image_id": key[0], "model": key[1]}
        vals = []
        for vlm_key in QIG_VLMS:
            v = scores.get(vlm_key)
            row[f"{vlm_key}_critic"] = round(v, 3) if v is not None else "n/a"
            if v is not None:
                vals.append(v)

        if vals:
            avg = st.mean(vals)
            std = st.pstdev(vals) if len(vals) > 1 else 0.0
            med = st.median(vals)
            row["average"] = round(avg, 3)
            row["std_dev"] = round(std, 3)
            row["median"] = round(med, 3)
            row["ensemble (mean \u00b1 std)"] = f"{avg:.3f} \u00b1 {std:.3f}"
        else:
            row["average"] = row["std_dev"] = row["median"] = row["ensemble (mean \u00b1 std)"] = "n/a"
        row["n_critics_scored"] = len(vals)
        rows.append(row)

    return pd.DataFrame(rows)


def main():
    tasks = build_tasks()
    if not tasks:
        print("No tasks to run -- fix the unresolved paths above first.")
        return

    all_scores = run_qig_inference(tasks)
    df = aggregate(tasks, all_scores)

    df.to_csv(OUT_CSV, index=False)
    print(f"\nWrote {OUT_CSV}")
    print(df.to_string())


if __name__ == "__main__":
    main()