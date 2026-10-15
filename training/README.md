# Fine-Tuning the Image Generator

This directory is dedicated to the domain adaptation of the BioVisGen Image Generator. The Image Generator is built upon [OmniGen2](https://github.com/VectorSpaceLab/OmniGen2) and is adapted to educational biology visualization through parameter-efficient fine-tuning on BioKosh, a collection of biological images paired with three levels of textual descriptions (Basic, Intermediate, and Advanced).

## Approach

Low-Rank Adaptation (LoRA) is employed, in which trainable low-rank adapter modules are introduced into the attention projection layers of the image generation module. The pretrained multi-modal backbone and the image tokenizer remain frozen. This enables efficient domain adaptation at a reduced computational cost and helps preserve the general multi-modal capabilities acquired during large-scale pretraining, thereby mitigating catastrophic forgetting.

## Configuration

The settings used for the experiments reported in the paper are summarized below.

| Parameter | Value |
|---|---|
| Training resolution | 768 × 768 × 3 |
| Optimizer / scheduler | AdamW / constant |
| Learning rate | 1 × 10⁻⁵ |
| Global batch size | 16 |
| LoRA rank | 16 |
| Mixed precision | bf16 |
| Maximum epochs | 100, with early stopping on validation loss (patience 15) |
| Data split | 80–20 train–test, with 5% of the training set held out for validation; Intermediate-level descriptions |

Experiments were conducted on a single NVIDIA RTX 6000 Ada Generation GPU (48 GB VRAM). Fine-tuning is performed on image-text pairs, where each image is paired with the description at the selected complexity level.

## Using a Fine-Tuned Checkpoint

The resulting LoRA adapter is loaded by the Image Generator at inference time. After fine-tuning, specify the checkpoint location in `biovisgen_agentic_architecture/config.py` or through the corresponding environment variable:

```bash
export LORA_PATH=/path/to/biokosh_lora/checkpoint/lora
```

Model checkpoints are not tracked in version control; see the main [README](../README.md) for the full list of configurable model paths.
