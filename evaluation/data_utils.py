"""
Data loading + flexible image path resolution.

Handles the fact that filenames share image_id but extensions may differ
(png/jpg/jpeg/webp).
"""

import csv
import glob
import os
from typing import Dict, List

IMAGE_EXTS = ("png", "jpg", "jpeg", "webp", "bmp", "tif", "tiff")


def load_descriptions(csv_path: str) -> Dict[str, str]:
    """Returns {image_id: description}."""
    out = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if "image_id" not in reader.fieldnames or "description" not in reader.fieldnames:
            raise ValueError(
                f"descriptions CSV must have 'image_id' and 'description' columns, "
                f"found: {reader.fieldnames}"
            )
        for row in reader:
            image_id = row["image_id"].strip()
            out[image_id] = row["description"].strip()
    return out


def _glob_by_stem(folder: str, stem: str) -> str:
    matches: List[str] = []
    for ext in IMAGE_EXTS:
        matches.extend(glob.glob(os.path.join(folder, f"{stem}.{ext}")))
        matches.extend(glob.glob(os.path.join(folder, f"{stem}.{ext.upper()}")))
    matches = sorted(set(matches))
    if len(matches) == 0:
        raise FileNotFoundError(f"No image found for stem '{stem}' in {folder}")
    if len(matches) > 1:
        raise FileNotFoundError(
            f"Multiple images found for stem '{stem}' in {folder}: {matches}. "
            f"Resolve ambiguity before running."
        )
    return matches[0]


def resolve_reference_image(images_root: str, ref_subfolder: str, image_id: str) -> str:
    folder = os.path.join(images_root, ref_subfolder)
    return _glob_by_stem(folder, image_id)


def resolve_generated_image(images_root: str, gen_model_cfg: dict, image_id: str) -> str:
    """
    gen_model_cfg is one value from config.GEN_MODELS, e.g.
      {"folder": "sdxl"} -> images_root/sdxl/{image_id}.*
    """
    folder = os.path.join(images_root, gen_model_cfg["folder"])
    stem = image_id + gen_model_cfg.get("suffix", "")
    return _glob_by_stem(folder, stem)


def build_task_list(images_root: str, descriptions: Dict[str, str], gen_models: Dict[str, dict],
                     ref_subfolder: str) -> List[dict]:
    """
    Returns a flat list of evaluation tasks:
      {image_id, model, description, ref_image_path, gen_image_path}
    Skips + reports any image_id/model combo whose file can't be resolved,
    rather than crashing the whole run.
    """
    tasks = []
    errors = []
    for image_id, description in descriptions.items():
        try:
            ref_path = resolve_reference_image(images_root, ref_subfolder, image_id)
        except FileNotFoundError as e:
            errors.append(str(e))
            continue

        for model_label, cfg in gen_models.items():
            try:
                gen_path = resolve_generated_image(images_root, cfg, image_id)
            except FileNotFoundError as e:
                errors.append(str(e))
                continue
            tasks.append({
                "image_id": image_id,
                "model": model_label,
                "description": description,
                "ref_image_path": ref_path,
                "gen_image_path": gen_path,
            })

    if errors:
        print(f"[data_utils] WARNING: {len(errors)} file(s) could not be resolved and were skipped:")
        for e in errors[:20]:
            print(f"  - {e}")
        if len(errors) > 20:
            print(f"  ... and {len(errors) - 20} more")

    return tasks
