# Archimedes / Math Explanation Generator

A fine-tuned FLAN-T5 transformer that generates step-by-step explanations for math problems. This repository contains the training and inference pipelines, data utilities, and a Typer-based CLI for training, evaluating, and interactively querying the model.

Key goals:
- Produce clear, human-readable step-by-step math explanations.
- Use parameter-efficient fine-tuning (LoRA) on top of google/flan-t5-base.
- Provide simple CLI tools for training and inference.

---

## Features
- LoRA fine-tuning of `google/flan-t5-base` on MathQA/GSM8K-style datasets.
- Training and evaluation pipelines using the Hugging Face `transformers` + `datasets` ecosystems.
- Lightweight terminal CLI (`typer`) for training, evaluation, and interactive inference.
- Utilities for tokenization, batching, and generation.

---

## Repository layout

Top-level:
- `app/` - application code (CLI, training, inference, data utils, model helpers)
- `outputs/` - default place to save trained models, checkpoints, logs
- `README.md` - this file
- `requirements.txt` - pinned Python dependencies

Inside `app/` (overview):
- `cli.py` - entrypoint for the Typer CLI (`python -m app.cli`)
- `train.py` - training loop and save logic
- `infer.py` (or `inference.py`) - inference helper(s)
- `data.py` - dataset loading and preprocessing helpers
- `model.py` - model + tokenizer loading helper
- `utils.py` - collate function and generation helper

---

## Model & Datasets

- Base model: `google/flan-t5-base` (Hugging Face).
- Fine-tuning method: LoRA (parameter-efficient fine-tuning) — the project supports applying LoRA during training.
- Example dataset: `miike-ai/mathqa` (MathQA) or similar arithmetic reasoning datasets. The `datasets` library is used to load them.

Dataset fields expected by the code:
- `question` (string) — the problem prompt
- `answer` (string) — the target explanation/solution text

---

## Setup

1. Create and activate a virtual environment

Windows (PowerShell):

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1
```

Windows (cmd.exe):

```cmd
python -m venv .venv && .\.venv\Scripts\activate
```

macOS / Linux (bash/zsh):

```bash
python -m venv .venv; source .venv/bin/activate
```

2. Install dependencies

```powershell
pip install -r requirements.txt
```

3. (Optional) If you plan to train with LoRA and use 8-bit loading, follow the `peft` and `bitsandbytes` installation notes in `requirements.txt` or the project docs. A GPU is strongly recommended for training.

---

## Quick usage (CLI)

All CLI commands are executed with the Typer entrypoint:

```powershell
python -m app.cli <command> [OPTIONS]
```

Commands supported (examples):

1) Training

Flag-style (explicit options):

```powershell
python -m app.cli train --model-name google/flan-t5-base --dataset-id miike-ai/mathqa --output-dir outputs/flan_t5_lora --epochs 3 --batch-size 4 --lora-r 8 --use-8bit
```

Positional-style (shorthand positional args accepted):

```powershell
python -m app.cli train google/flan-t5-base miike-ai/mathqa outputs/flan_t5_lora --epochs 3 --batch-size 4
```

Notes:
- `--use-8bit` is a flag (no value) to enable 8-bit model loading/training when supported.
- `--lora-r` controls the LoRA rank. Adjust to your needs.
- Output directory is where the fine-tuned model/tokenizer will be saved.

2) Inference (single-shot)

Flag-style:

```powershell
python -m app.cli infer --model-dir outputs/flan_t5_lora --question "What is 2+2?"
```

Positional-style:

```powershell
python -m app.cli infer outputs/flan_t5_lora "What is 2+2?"
```

3) Interactive inference

There is also an interactive CLI mode (if implemented in `app/cli.py`) that runs a REPL allowing you to ask multiple questions in the same session:

```powershell
python -m app.cli infer
```

Type `exit` or `quit` to leave the interactive session.

---

## Example walkthrough

Problem: "Robert sold 60 apples to his classmates in June, and then he sold two-thirds as many apples in July. How many apples did Robert sell altogether in June and July?"

How the model should reason (human-readable expected steps):
1. Robert sold 60 apples in June.
2. In July he sold two-thirds as many as in June: 2/3 * 60 = 40.
3. Total sold = June + July = 60 + 40 = 100 apples.

Expected final answer: 100

This repository aims to train the model to produce step-by-step reasoning like the outline above, followed by the final numeric answer.

---

## Hardware & performance notes

- Training: GPU strongly recommended. LoRA reduces memory footprint but training on CPU will be slow.
- Inference: CPU can be used for lightweight usage, but GPU gives faster generation and enables larger batch sizes.
- If you enable `--use-8bit`, make sure `bitsandbytes` and compatible toolchains are installed and your environment supports 8-bit loading.

---

## Troubleshooting

- If the CLI fails with "unexpected extra arguments", ensure you are running the latest `app/cli.py` from this repo; the CLI accepts both positional and flag-style invocations.
- If model loading fails due to missing packages, install packages from `requirements.txt` and check your PyTorch/transformers installation.
- If dataset downloading fails, ensure you have an internet connection and valid `datasets` credentials where necessary.

---

## Development notes & next steps

- Add evaluation metrics (exact match, BLEU, or human-evaluated rubric) and automated tests.
- Add a small sample dataset and unit tests for data preprocessing and the collate function.
- Add docker setup for reproducible environments.

---

If anything in this README doesn't match your local code (file names, CLI entrypoint, or command names), update the corresponding files in `app/` or open an issue describing the mismatch.
