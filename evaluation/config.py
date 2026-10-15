"""
Central configuration for the VLM ensemble evaluation pipeline.
Edit the paths and GEN_MODELS block to match your exact folder layout.
"""

# ---------------------------------------------------------------------------
# Paths (edit these)
# ---------------------------------------------------------------------------
IMAGES_ROOT = "images"                     # contains subfolders per generation model + "reference"
DESCRIPTIONS_CSV = "descriptions.csv"      # columns: image_id, description
REFERENCE_SUBFOLDER = "reference"

# ---------------------------------------------------------------------------
# Generation models under evaluation
# ---------------------------------------------------------------------------
# Each entry is a "model" label that will appear in the `model` column of the
# output CSV. `folder` is the subfolder under IMAGES_ROOT to glob for the
# image (image_id.* -> extension may vary, handled in data_utils.py).

GEN_MODELS = {
    "stable_diffusion": {"folder": "sd"},
    "sdxl": {"folder": "sdxl"},
    "galip": {"folder": "galip"},
    "omnigen": {"folder": "omnigen"},
    "omnigen2": {"folder": "omnigen2"},
}
# 5 generation models x 50 images = 250 (image_id, model) tasks per VLM.

# ---------------------------------------------------------------------------
# VLM ensemble registry
# ---------------------------------------------------------------------------
VLM_REGISTRY = {
    "qwen3vl": {
        "hf_id": "Qwen/Qwen3-VL-8B-Instruct",
        "module": "models.qwen3vl_eval",
        "class": "Qwen3VLEvaluator",
    },
    "internvl": {
        "hf_id": "OpenGVLab/InternVL3_5-8B-HF",
        "module": "models.internvl_eval",
        "class": "InternVLEvaluator",
    },
    # "minicpm": {
    #     "hf_id": "openbmb/MiniCPM-V-4_5",
    #     "module": "models.minicpm_eval",
    #     "class": "MiniCPMEvaluator",
    # },
    "deepseekvl": {
       "hf_id": "deepseek-community/deepseek-vl-7b-chat",
       "module": "models.deepseekvl_eval",
       "class": "DeepseekVLEvaluator",
   },
    "gemma4": {
        "hf_id": "google/gemma-4-12B-it",
        "module": "models.gemma4_eval",
        "class": "Gemma4Evaluator",
    },
}

# ---------------------------------------------------------------------------
# Generation params
# ---------------------------------------------------------------------------
MAX_NEW_TOKENS = 512
TEMPERATURE = 0.2   # low temperature for scoring consistency
MAX_RETRIES = 3      # retries on malformed JSON output
