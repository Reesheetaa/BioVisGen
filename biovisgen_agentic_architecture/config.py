import os

# =========================
# MODEL CONFIG
# =========================

BASE_MODEL = os.getenv("OMNIGEN_BASE_MODEL", "OmniGen2/OmniGen2")

# Replace these placeholders with local paths after obtaining/training the models.
OMNIGEN_REPO_PATH = os.getenv("OMNIGEN_REPO_PATH", "/path/to/OmniGen2")

LORA_PATH = os.getenv("LORA_PATH", "/path/to/biology17k_lora/checkpoint/lora")
SDXL_INPAINTING_MODEL = os.getenv(
    "SDXL_INPAINTING_MODEL", "diffusers/stable-diffusion-xl-1.0-inpainting-0.1"
)
SDXL_CHECKPOINT_PATH = os.getenv("SDXL_CHECKPOINT_PATH", "/path/to/your/sdxl-inpainting-checkpoint")

# =========================
# VLM CRITIC
# =========================

VLM_MODEL = os.getenv("BIOVISGEN_CRITIC_MODEL", "Qwen/Qwen2.5-VL-3B-Instruct")

# =========================
# DEVICE
# =========================

DEVICE = os.getenv("BIOVISGEN_DEVICE", "cuda")
DTYPE = os.getenv("BIOVISGEN_DTYPE", "bfloat16")

# =========================
# GENERATION SETTINGS
# =========================

HEIGHT = 768
WIDTH = 768

NUM_STEPS = 50

SEED = 42

# Initial generation
GUIDANCE_SCALE = 8.5

# OmniGen2 edit mode
IMAGE_GUIDANCE_SCALE = 2.0

# Maximum critic-driven edit rounds
MAX_REFINEMENT_ROUNDS = 3

# =========================
# CRITIC
# =========================

CRITIC_THRESHOLD = 0.67

# =========================
# OUTPUTS
# =========================

OUTPUT_DIR = os.getenv("BIOVISGEN_OUTPUT_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs"))

INTERMEDIATE_DIR = os.path.join(
    OUTPUT_DIR,
    "intermediate"
)

FINAL_DIR = os.path.join(
    OUTPUT_DIR,
    "final"
)

os.makedirs(
    INTERMEDIATE_DIR,
    exist_ok=True
)

os.makedirs(
    FINAL_DIR,
    exist_ok=True
)