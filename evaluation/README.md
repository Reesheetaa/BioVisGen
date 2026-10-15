# SEAS Evaluation Pipeline

This directory contains the implementation of **Simulated Expert Assessment Scoring (SEAS)**, the VLM-based protocol used to assess the pedagogical and semantic quality of generated educational biology diagrams. Each generated image is scored on three criteria, **scientific accuracy**, **visual clarity**, and **complexity alignment**, on a 10-point scale. The overall score is computed deterministically in code as the mean of the three criterion scores; the VLM is never asked to compute it.

Scoring is performed by an ensemble of three VLMs (**QIG**):

| Key | Model |
|---|---|
| `qwen3vl` | [Qwen3-VL-8B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct) |
| `internvl` | [InternVL3.5-8B](https://huggingface.co/OpenGVLab/InternVL3_5-8B-HF) |
| `gemma4` | [Gemma-4-12B-it](https://huggingface.co/google/gemma-4-12B-it) |

The pipeline supports two modes of use: **batch evaluation** over a set of samples and generation models, and **evaluation of a single sample**. In both cases, the generated image is judged primarily against the input description, with the reference image provided for contextual comparison only, and the evaluation focuses on structural and conceptual correctness rather than visual style.

---

## 1. Installation

```bash
pip install -U "transformers>=4.57" torch torchvision accelerate pillow scipy
```

InternVL requires `trust_remote_code`; no separate installation is needed, as the modeling code is fetched from the Hugging Face Hub. If model loading fails with an unrecognized model type (this may occur for Qwen3-VL or Gemma-4 with older `transformers` versions), install `transformers` from source:

```bash
pip install -U git+https://github.com/huggingface/transformers
```

Model weights are downloaded from Hugging Face on first use and may require authentication and acceptance of the corresponding licenses. Each 8B model fits comfortably within a single 48 GB GPU.

---

## 2. Evaluating a Single Sample

A single generated image can be scored without constructing the full dataset layout. The evaluator requires the textual description, the reference image (used for context only), and the generated image. From within `evaluation/`:

```python
import config
from run_evaluation import load_evaluator
from prompts import parse_vlm_json

description = "This diagram illustrates the human digestive system ..."
reference   = "path/to/reference.png"
generated   = "path/to/generated.png"

scores = {}
for vlm in ["qwen3vl", "internvl", "gemma4"]:          # QIG ensemble
    evaluator = load_evaluator(vlm, gpu=0)
    raw = evaluator.evaluate(
        description, reference, generated,
        max_new_tokens=config.MAX_NEW_TOKENS,
        temperature=config.TEMPERATURE,
    )
    scores[vlm] = parse_vlm_json(raw)   # scientific_accuracy, visual_clarity,
                                        # complexity_alignment, overall_score, reasoning

es = sum(s["overall_score"] for s in scores.values()) / len(scores)
print(f"SEAS ensemble score (Es): {es:.2f}")
```

Each VLM returns the three criterion scores together with a textual justification (`reasoning`). A single VLM can be used by restricting the list of keys; the ensemble score is the mean over the VLMs that are run. Responses that cannot be parsed raise `ResponseParseError`; evaluation can simply be retried, as in the batch pipeline (up to `config.MAX_RETRIES`).

---

## 3. Batch Evaluation

### 3.1 Data layout

Set `IMAGES_ROOT` and `DESCRIPTIONS_CSV` in `config.py` to the local dataset directory. `IMAGES_ROOT` should contain one flat folder per generation model, in addition to the reference images, with files named by image identifier:

```
images/
├── reference/
├── sd/
├── sdxl/
├── galip/
├── omnigen/
└── omnigen2/
descriptions.csv        # columns: image_id, description
```

The generation models under evaluation are defined in `config.py` (`GEN_MODELS`). For 50 samples and five generation models, this yields 250 (image, model) evaluation tasks per VLM, or 750 VLM calls across the ensemble.

### 3.2 Running the ensemble

Each VLM is run once and writes its own CSV. Runs for different VLMs can be executed in parallel on separate GPUs; choose GPU indices that exist on your machine.

```bash
python run_evaluation.py --model qwen3vl  --gpu 0 --output results_qwen3vl.csv
python run_evaluation.py --model internvl --gpu 1 --output results_internvl.csv
python run_evaluation.py --model gemma4   --gpu 0 --output results_gemma4.csv
```

Runs are **resumable**: if interrupted, re-run the same command and completed (image, model) pairs are skipped. Unparseable VLM responses are logged to `<output>.errors.log` and retried up to `config.MAX_RETRIES` times before being skipped; re-running the script attempts the skipped rows again. Each CSV has the columns:

```
image_id, model, scientific_accuracy, visual_clarity, complexity_alignment, overall_score, reasoning, vlm
```

### 3.3 Aggregation

```bash
python aggregate_ensemble.py \
    --vlm qwen3vl=results_qwen3vl.csv \
    --vlm internvl=results_internvl.csv \
    --vlm gemma4=results_gemma4.csv \
    --out-detail ensemble_detail.csv \
    --out-summary ensemble_summary.csv
```

- `ensemble_detail.csv`: one row per (image, generation model), with each VLM's scores as separate columns, the ensemble mean per criterion and overall, and `n_vlms_scored`, which flags rows where a VLM failed to produce a score.
- `ensemble_summary.csv`: mean and population standard deviation of the ensemble scores per generation model across all samples.

### 3.4 Supporting utilities

| Script | Purpose |
|---|---|
| `check_missing.py` | Reports (image, model) evaluations still missing for each VLM's results CSV |
| `recover_from_errors_log.py` | Re-parses logged raw responses without re-running the VLM and appends recoverable rows |
| `aggregate_critics.py` | One column per VLM with row-wise average, standard deviation, and median |
| `level-wise-seas.py` | Computes SEAS for models fine-tuned on the Basic and Advanced description levels; metadata and checkpoint paths must be edited to match the local setup |

---

## 4. Evaluation Prompt

The rubric is defined in `prompts.py`. It instructs the VLM to treat the description as the primary basis for assessment, the reference image as contextual only, and to focus on structural and conceptual correctness rather than style, with detailed sub-criteria for each of the three criteria. Sampling is performed at a temperature of 0.2 (`config.TEMPERATURE`) without a fixed seed, so scores may vary slightly across runs.
