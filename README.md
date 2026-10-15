# BioVisGen: An Agentic Multi-Modal Framework for Visualization of Biology Concepts

Rishita Patel, Sushree S. Behera, Srinath Srinivasa, Prasad Ram

This repository provides the architecture code of **BioVisGen**, an agentic multi-modal framework for generating educational biology diagrams from textual descriptions, together with the evaluation pipeline used in the accompanying paper. The framework is model-agnostic at the configuration level: the individual model components, including the fine-tuned Image Generator, are loaded from user-specified paths and identifiers (see [Installation](#installation) and [Usage](#usage)).

---

## Overview

Personalized learning requires educational resources that adapt in both content complexity and mode of presentation. Creating such resources manually demands substantial time and effort, which limits scalability. BioVisGen addresses this bottleneck by allowing instructors to describe a biological concept in natural language and obtain a pedagogically appropriate diagram, which can then be evaluated, corrected, and refined with minimal manual intervention.

The framework integrates specialized agents for prompt structuring, image generation, output evaluation, and refinement, coordinated by an Orchestrator. The Image Generator is built upon [OmniGen2](https://github.com/VectorSpaceLab/OmniGen2) and adapted to the biology domain through LoRA fine-tuning on **BioKosh**, an in-house dataset of 17,000 biological images paired with three levels of textual descriptions (Basic, Intermediate, and Advanced). In addition, the paper proposes an education-specific evaluation protocol, **Simulated Expert Assessment Scoring (SEAS)**, which scores generated diagrams on scientific accuracy, visual clarity, and complexity alignment using a VLM ensemble.

**Key contributions**

- **BioVisGen**, an agentic framework that combines prompt structuring, image generation, critic-guided evaluation, and human-in-the-loop refinement.
- **BioKosh**, a K–12 biology dataset of 17,000 images with multi-level textual descriptions (51,000 image-text pairs).
- **Human-in-the-loop post-processing** supporting text removal, mask-guided localized editing, and instruction-guided semantic modification.
- **SEAS**, an evaluation framework that uses a VLM ensemble to assess the pedagogical and semantic quality of generated educational diagrams.

---

## Framework

Given an input prompt, the Orchestrator autonomously selects the generation or refinement pathway based on the prompt, intermediate outputs, critic feedback, and user requests.

| Agent | Role | Implementation | Source |
|---|---|---|---|
| **Orchestrator** | Coordinates the workflow and invokes the other agents; assesses prompt quality (scientific structures, spatial relationships, educational presentation, visual organization, labeling requirements; 0–5) and routes among actions | Workflow control logic coordinating all agents | `agents/orchestrator.py` |
| **Prompt Refiner** | Rewrites ambiguous prompts into structured descriptions of components, relationships, and spatial organization | [Phi-3 Mini](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct) | `agents/prompt_refiner.py`, `agents/llm_agent.py` |
| **Image Generator** | Synthesizes the diagram; also performs instruction-guided semantic editing | [OmniGen2](https://github.com/VectorSpaceLab/OmniGen2) with LoRA adapters | `agents/image_generator.py` |
| **Critic** | Scores image-text alignment; when the score is below threshold, produces structure-only refinement instructions | [CLIP](https://github.com/openai/CLIP) ViT-B/32 (via [OpenCLIP](https://github.com/mlfoundations/open_clip)); [Qwen2.5-VL-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct) | `agents/critic.py` |
| **Image Refiner** | Applies localized edits (text removal, mask-guided modification) without regenerating the full image | [EasyOCR](https://github.com/JaidedAI/EasyOCR) mask generator; [SDXL inpainting](https://huggingface.co/diffusers/stable-diffusion-xl-1.0-inpainting-0.1) | `agents/mask_generator.py`, `agents/sdxl_inpainter.py` |

During critic-guided refinement, the Critic computes a CLIP-based alignment score between the generated image and the conditioning prompt. If the score falls below the threshold, the Critic VLM describes structural deficiencies, and the Orchestrator passes these instructions together with the previous image to the Image Generator for iterative editing, for at most `MAX_REFINEMENT_ROUNDS` iterations. The highest-scoring image across all iterations is returned.

---

## Repository Structure

```
BioVisGen/
├── run.py                                # Entry point for the interactive (Gradio) system
├── .env.example                          # Environment variables for model paths and device
├── biovisgen_agentic_architecture/
│   ├── main.py                           # Gradio interface
│   ├── config.py                         # Model paths and generation / critic settings
│   ├── prompts.py                        # Evaluation prompts used by the batch script
│   ├── batch_run.py                      # Batch generation with critic-guided refinement
│   ├── test_refiner.py                   # Smoke test for the Prompt Refiner
│   ├── requirements.txt
│   ├── agents/                           # Orchestrator, Prompt Refiner, Image Generator,
│   │                                     # Critic, Mask Generator, SDXL Inpainter
│   └── utils/
├── evaluation/                           # SEAS evaluation pipeline (VLM ensemble)
└── training/                             # Model fine-tuning
```

---

## Installation

```bash
git clone <repository-url>
cd BioVisGen

python -m venv .venv
source .venv/bin/activate          # Windows PowerShell: .venv\Scripts\Activate.ps1

pip install -r biovisgen_agentic_architecture/requirements.txt
```

**OmniGen2.** OmniGen2 is an external codebase and is not vendored in this repository. Clone and install it separately following its instructions ([VectorSpaceLab/OmniGen2](https://github.com/VectorSpaceLab/OmniGen2)), then point `OMNIGEN_REPO_PATH` to the local clone.

**Model weights.** This repository contains the architecture code; the model components it orchestrates are supplied by the user, either through local paths or through Hugging Face identifiers. Where a component is not already available locally, the following are downloaded from Hugging Face on first use and may require authentication and acceptance of the corresponding licenses: [OmniGen2](https://huggingface.co/OmniGen2/OmniGen2) (`OmniGen2/OmniGen2`), [Phi-3 Mini](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct) (`microsoft/Phi-3-mini-4k-instruct`), [Qwen2.5-VL-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct), the CLIP ViT-B/32 weights loaded through [OpenCLIP](https://github.com/mlfoundations/open_clip), the [EasyOCR](https://github.com/JaidedAI/EasyOCR) detection models, and the [SDXL inpainting pipeline](https://huggingface.co/diffusers/stable-diffusion-xl-1.0-inpainting-0.1) (`diffusers/stable-diffusion-xl-1.0-inpainting-0.1`). The BioKosh-adapted LoRA checkpoint is specified through `LORA_PATH` (see [Training](#training)).

### Configuration

Paths and device settings are read from environment variables (see `.env.example`) or set directly in `biovisgen_agentic_architecture/config.py`.

```bash
export OMNIGEN_REPO_PATH=/path/to/OmniGen2
export LORA_PATH=/path/to/biokosh_lora/checkpoint/lora
export SDXL_CHECKPOINT_PATH=/path/to/sdxl-inpainting-checkpoint   # optional; falls back to the Hub model ID
export BIOVISGEN_DEVICE=cuda
```

| Setting | Default | Description |
|---|---|---|
| `BASE_MODEL` | `OmniGen2/OmniGen2` | OmniGen2 base model identifier |
| `LORA_PATH` | placeholder | LoRA adapter fine-tuned on BioKosh |
| `VLM_MODEL` | `Qwen/Qwen2.5-VL-3B-Instruct` | Critic VLM for refinement instructions |
| `HEIGHT` × `WIDTH` | 768 × 768 | Output resolution |
| `NUM_STEPS` | 50 | Diffusion inference steps |
| `GUIDANCE_SCALE` / `IMAGE_GUIDANCE_SCALE` | 8.5 / 2.0 | Text guidance (generation and editing) / image guidance (editing) |
| `MAX_REFINEMENT_ROUNDS` | 3 | Maximum critic-guided regeneration limit *T* |
| `CRITIC_THRESHOLD` | 0.4 | Alignment threshold, applied to the rescaled CLIP score (cosine similarity + 1) / 2 |
| `SEED` | 42 | Random seed for generation and editing |

**Hardware.** The experiments in the paper were conducted on a single NVIDIA RTX 6000 Ada Generation GPU (48 GB VRAM). A CUDA-capable GPU is recommended. In the current release, Phi-3 Mini and the Qwen2.5-VL critic are instantiated on the CPU in float32 regardless of `BIOVISGEN_DEVICE`, whereas OmniGen2, CLIP, and the SDXL inpainter use the configured device.

---

## Usage

Before running any component, ensure that the model and checkpoint paths are configured, either through environment variables or directly in `biovisgen_agentic_architecture/config.py` (see [Configuration](#configuration)). In particular, `OMNIGEN_REPO_PATH` should point to the local OmniGen2 codebase, `LORA_PATH` to the LoRA checkpoint used by the Image Generator, and `SDXL_CHECKPOINT_PATH` to a local SDXL inpainting checkpoint (optional; the public model identifier is used otherwise). Components that are already configured, or available locally, require no further changes.

### Interactive system

```bash
python run.py
```

Open the local Gradio URL printed in the terminal (default port 7860). The interface exposes three Orchestrator actions:

1. **Generate a biology illustration.** The prompt is assessed and, if necessary, refined; OmniGen2 synthesizes the diagram; the Critic scores it and triggers iterative refinement when required. The refined prompt, final image, best critic score, and execution log are displayed.
2. **Remove text.** EasyOCR detects textual regions and constructs a binary mask, and the SDXL inpainter replaces the masked regions with a seamless background, leaving the remaining structure unchanged.
3. **Localized refinement.** The user uploads an image, draws a mask over the target region, and provides a modification prompt; the SDXL inpainter edits only the masked region.

Heavy agents are initialized lazily upon the first request. Generated images are written to `biovisgen_agentic_architecture/outputs/` (`intermediate/` for each iteration, `final/` for the selected output).

### Batch generation

```bash
cd biovisgen_agentic_architecture
python batch_run.py
```

This script runs the full prompt-assessment, generation, and critic-guided refinement loop over the Basic-level prompts defined in `prompts.py` and stores the results, including intermediate iterations, in a local output directory.

---

## BioKosh Dataset

BioKosh covers K–12 biology concepts derived from [NCERT](https://ncert.nic.in/textbook.php) textbooks. Unlike conventional text-vision datasets that provide a single caption per image, each image is annotated at three levels reflecting different learning proficiencies, and the source reference of each image is retained for traceability.

| Attribute | Description |
|---|---|
| Domain | K–12 biology: cellular structures, anatomy, physiological processes, and biological classifications |
| Size | 17,000 images × 3 description levels = 51,000 image-text pairs |
| Levels | **Basic** (concise, simple vocabulary); **Intermediate** (essential terminology, moderate detail); **Advanced** (comprehensive scientific detail and conceptual relationships) |
| Collection | Concepts identified from NCERT textbooks, followed by keyword-based web retrieval and manual selection of informative images; excessive text, logos, and watermarks removed |
| Exclusions | Images below 128 × 128 pixels, excessive blur, presentation slides, text-dominant images, and unstructured visualizations |
| Annotation | Three-level descriptions generated with [Claude 3.7 Sonnet](https://www.anthropic.com/news/claude-3-7-sonnet) using a structured meta-prompt that enforces grounding in the visual content |

**Availability and licensing.** To respect the intellectual property rights of the original materials, the images themselves are not redistributed. BioKosh is intended for academic and non-commercial research use only, and users must comply with the licenses and usage restrictions of the original image sources. Details regarding access will be provided with the final release.

---

## Training

The Image Generator is fine-tuned from the pretrained OmniGen2 model using Low-Rank Adaptation (LoRA), with adapters introduced into the attention projection layers of the image generation module while the multi-modal backbone and image tokenizer remain frozen. The configuration used in the paper is listed below to support reproduction; the associated fine-tuning scripts will accompany the final release.

| Parameter | Value |
|---|---|
| Training resolution | 768 × 768 × 3 |
| Optimizer / scheduler | AdamW / constant |
| Learning rate | 1 × 10⁻⁵ |
| Global batch size | 16 |
| LoRA rank | 16 |
| Mixed precision | bf16 |
| Maximum epochs | 100, with early stopping on validation loss (patience 15) |
| Data split | 80–20 train–test, with 5% of the training set held out for validation (12,920 / 680 / 3,400 image-text pairs); Intermediate-level descriptions |

After fine-tuning, set `LORA_PATH` to the resulting adapter checkpoint.

---

## Evaluation

Beyond CLIP-I (*C<sub>I</sub>*) and CLIP-T (*C<sub>T</sub>*) similarity, generated diagrams are assessed on three education-specific criteria: **scientific accuracy**, **visual clarity**, and **complexity alignment**. Each criterion is rated on a 10-point scale, and the overall score is the mean of the three.

**Simulated Expert Assessment Scoring (SEAS).** Scoring is performed by an ensemble of three VLMs (**QIG**): [Qwen3-VL-8B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct), [InternVL3.5-8B](https://huggingface.co/OpenGVLab/InternVL3_5-8B-HF), and [Gemma-4-12B-it](https://huggingface.co/google/gemma-4-12B-it). Each VLM independently evaluates the same set of generated samples against the input description, with the reference image provided for contextual comparison only, and returns criterion-wise scores with textual reasoning. Sampling is performed at a temperature of 0.2. The final score is the mean of the individual VLM scores, and inter-evaluator reliability is reported using the Intraclass Correlation Coefficient ICC(C,k).

### Running the QIG ensemble

```bash
cd evaluation

python run_evaluation.py --model qwen3vl  --gpu 0 --output results_qwen3vl.csv
python run_evaluation.py --model internvl --gpu 0 --output results_internvl.csv
python run_evaluation.py --model gemma4   --gpu 0 --output results_gemma4.csv

python aggregate_ensemble.py \
    --vlm qwen3vl=results_qwen3vl.csv \
    --vlm internvl=results_internvl.csv \
    --vlm gemma4=results_gemma4.csv \
    --out-detail ensemble_detail.csv \
    --out-summary ensemble_summary.csv
```

Runs are resumable: completed (image, model) pairs are skipped on re-execution, and unparseable responses are logged to `<output>.errors.log`. Supporting utilities include `check_missing.py` (report missing evaluations), `recover_from_errors_log.py` (re-parse logged responses), and `aggregate_critics.py` (per-critic scores with mean, standard deviation, and median). `level-wise-seas.py` computes SEAS for models fine-tuned on the Basic and Advanced description levels; its metadata and checkpoint paths must be edited to match the local setup.

---

## Summary of Results

Detailed results, qualitative comparisons, and ablations are reported in the paper. Selected findings are summarized below.

**Comparison of fine-tuned image generators** (BioKosh test set, Intermediate-level descriptions)

| Model | Train (s/step) | Inference (s/image) | *C<sub>I</sub>* ↑ | *C<sub>T</sub>* ↑ |
|---|---|---|---|---|
| **[OmniGen2](https://github.com/VectorSpaceLab/OmniGen2)** (Image Generator) | 11.80 | 23.56 | 0.6905 ± 0.0951 | **0.3376 ± 0.0317** |
| [OmniGen](https://github.com/VectorSpaceLab/OmniGen) | 7.10 | 21.17 | 0.6608 ± 0.0977 | 0.3074 ± 0.0403 |
| [SDXL](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0) | 3.70 | 4.87 | **0.7255 ± 0.0834** | 0.3273 ± 0.0304 |
| [Stable Diffusion](https://github.com/CompVis/stable-diffusion) | 0.85 | 3.07 | 0.6673 ± 0.0988 | 0.3159 ± 0.0331 |
| [GALIP](https://github.com/tobran/GALIP) | 0.181 | 0.045 | 0.7210 ± 0.0790 | 0.3247 ± 0.0279 |

OmniGen2 attains the highest *C<sub>T</sub>* and the highest SEAS score (5.47, against 3.38–4.89 for the other models), at the cost of greater training and inference time. Because CLIP is pretrained on generic image-text data, *C<sub>I</sub>* and *C<sub>T</sub>* only partially capture the structural and pedagogical qualities of biological diagrams.

**Evaluation consistency.** The SEAS ensemble exhibits moderate inter-evaluator agreement (ICC(C,k) = 0.5086), which may reflect the diversity and complexity of the generated outputs.

---

## Reproducibility Notes

- Generation and editing use a fixed seed (`SEED = 42`); SEAS evaluation is sampled at temperature 0.2 without a fixed seed, so scores may vary slightly across runs.
- Model versions, hardware, and library versions affect results; please verify the versions of PyTorch, Transformers, Diffusers, and OmniGen2 used in your environment.
- Model checkpoints and result files are not bundled with this repository.

---

## Citation

The associated paper has been accepted, and its citation details (including DOI) will be added to this repository upon publication. This repository is currently provided for review purposes.

Please also cite the original works for OmniGen2, SDXL, CLIP, EasyOCR, Phi-3, Qwen2.5-VL, and the VLMs used in the evaluation ensemble.

---

## License

The code, documentation, and any original material in this repository are released under the [Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International (CC BY-NC-ND 4.0)](https://creativecommons.org/licenses/by-nc-nd/4.0/) license.

This license applies in addition to, and does not replace, the licenses of third-party components. Use of the following is subject to their respective terms: [OmniGen2](https://github.com/VectorSpaceLab/OmniGen2), [Stable Diffusion XL inpainting](https://huggingface.co/diffusers/stable-diffusion-xl-1.0-inpainting-0.1), [EasyOCR](https://github.com/JaidedAI/EasyOCR), [CLIP](https://github.com/openai/CLIP), [Phi-3 Mini](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct), [Qwen2.5-VL](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct), [Qwen3-VL](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct), [InternVL3.5](https://huggingface.co/OpenGVLab/InternVL3_5-8B-HF), and [Gemma](https://huggingface.co/google/gemma-4-12B-it). BioKosh is additionally subject to the licenses of the original image sources (see [BioKosh Dataset](#biokosh-dataset)).
