# RLUF: Reinforcement Learning with Uncertainty Feedback

[![PyPI version](https://badge.fury.io/py/conformal-feedback-alignment.svg)](https://pypi.org/project/conformal-feedback-alignment/)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Conformal Feedback Alignment — using **Conformal Prediction** to quantify LLM generation uncertainty and integrate it as per-example weights into **DPO** training. This enables the model to learn more from high-confidence preference pairs and less from uncertain ones.

> **Paper:** [Conformal Feedback Alignment: Quantifying Answer-Level Reliability for Robust LLM Alignment](https://aclanthology.org/2026.findings-eacl.184/) (Findings of EACL 2026)

## Installation

```bash
pip install conformal-feedback-alignment
```

Or install from source for development:

```bash
git clone https://github.com/tiejin98/Conformal-Feedback-Alignment.git
cd Conformal-Feedback-Alignment
pip install -e ".[dev]"
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    RLUF Pipeline                            │
│                                                             │
│  ┌──────┐    ┌───────────-┐    ┌──────────────────┐         │
│  │ SFT  │───>│ Generation │───>│ Conformal        │         │
│  │      │    │ + Scoring  │    │ Prediction (CP)  │         │
│  └──────┘    └───────────—┘    └────────┬─────────┘         │
│                                        │                    │
│                              Prediction Sets                │
│                              (50% & 80% coverage)           │
│                                        │                    │
│  ┌───────────────┐    ┌───────────────▼────────────—──┐     │
│  │ AI Feedback   │───>│ Uncertainty Weight Assignment │     │
│  │ (Preference)  │    └───────────────┬─────────────—─┘     │
│  └───────────────┘                    │                     │
│                              Weighted DPO Pairs             │
│                                        │                    │
│  ┌──────────────┐    ┌────────────────▼─────┐               │
│  │ Weighted DPO │───>│ Inference + Evaluate │               │
│  │ Training     │    └──────────────────────┘               │
│  └──────────────┘                                           │
└─────────────────────────────────────────────────────────────┘
```

## Quick Start

The `Quick Start/` folder provides pre-processed sample data so you can **skip Stages 1–2** and directly run training and evaluation. All files are based on **Llama-2-7b** on the **Summarization** task.

We provide three categories of files: the **training data** (already weighted by conformal uncertainty), the **generated outputs** from both RLUF and base DPO models on the test set, and the **evaluation scores** from GPT-4o — so you can either retrain from scratch or directly inspect the results without running any inference.

| File | Description |
|------|-------------|
| `dpo_data_llama2_withuncertainty.zip` | DPO training data (104,614 examples) with `prompt`, `chosen`, `rejected`, and `weight` fields — extract before use |
| `test_dict_question.pkl` | 639 test-set questions |
| `test_dict_RLUF.pkl` | Model outputs from the RLUF-trained model |
| `test_dict_baseDPO.pkl` | Model outputs from the base DPO model (for comparison) |
| `evaluation_scores_RLUF.pkl` | Pre-computed GPT-4o evaluation scores for RLUF |
| `evaluation_scores_baseDPO.pkl` | Pre-computed GPT-4o evaluation scores for base DPO |

### Option A: Using Standalone Scripts

All scripts read API keys from the `.env` file in the project root. Set it up first:

```bash
cp .env.example .env
# Edit .env: OPENAI_API_KEY=sk-..., HF_TOKEN=hf_...
```

Then run training and evaluation directly:

```bash
cd "Quick Start"
unzip dpo_data_llama2_withuncertainty.zip

# Step 1: Train
python dpo_ours_train.py

# Step 2: Evaluate (calls GPT-4o, requires OPENAI_API_KEY)
python AI_response_evaluation.py

# Step 3: Read pre-computed scores (no API key needed)
python read_evaluation.py
```

### Option B: Using the `cfa` CLI

Set up the output directory structure that `cfa` expects, then use the CLI:

```bash
pip install -e .

# Set up API keys
cp .env.example .env
# Edit .env: OPENAI_API_KEY=sk-..., HF_TOKEN=hf_...

# Prepare data in cfa output structure
mkdir -p outputs/feedback outputs/inference outputs/evaluation
cd "Quick Start"
unzip dpo_data_llama2_withuncertainty.zip
cp dpo_data_llama2_withuncertainty.json ../outputs/feedback/
cp test_dict_question.pkl test_dict_RLUF.pkl test_dict_baseDPO.pkl ../outputs/inference/
cp evaluation_scores_RLUF.pkl evaluation_scores_baseDPO.pkl ../outputs/evaluation/
cd ..

# Train
cfa train --config configs/default.yaml

# Evaluate (or skip if using the pre-computed scores above)
cfa evaluate --config configs/default.yaml
```

### Expected Results

The pre-computed scores in the provided `.pkl` files reproduce the following results:

| Model | Acc | Rel | Comp | Expr | Overall |
|-------|-----|-----|------|------|---------|
| Base DPO | 6.359 | 7.319 | 5.351 | 7.144 | 6.543 |
| **RLUF (Ours)** | **6.462** | **7.547** | **5.391** | **7.421** | **6.705** |

> **Note:** The paper reports Overall scores of 65.68 (Base DPO) vs. 67.30 (RLUF). The numbers above are from a re-run with updated evaluation (GPT-4o), so the absolute values differ slightly, but the relative improvement of RLUF over Base DPO is consistent.

## Full Pipeline

To run the full pipeline from scratch (all three stages), configure your environment and use the `cfa` CLI:

```bash
# 1. Set up your API keys
cp .env.example .env
# Edit .env: OPENAI_API_KEY=sk-..., HF_TOKEN=hf_...

# 2. Run individual stages:
cfa sft --config configs/default.yaml          # Stage 1a: SFT fine-tuning
cfa generate --config configs/default.yaml     # Stage 1b: Multi-sample generation + GPT scoring
cfa calibrate --config configs/default.yaml    # Stage 1c: Conformal prediction calibration
cfa feedback --config configs/default.yaml     # Stage 2a: Pairwise preference annotation
cfa assign-weights --config configs/default.yaml  # Stage 2b: Assign uncertainty weights
cfa train --config configs/default.yaml        # Stage 3a: Weighted DPO training
cfa infer --config configs/default.yaml        # Stage 3b: Test set inference
cfa evaluate --config configs/default.yaml     # Stage 3c: GPT-4o evaluation scoring

# Or run everything at once:
cfa run-all --config configs/default.yaml
```

You can also run unit tests to verify the utility functions:

```bash
pytest tests/ -v
```

This runs tests for CP scoring, text processing, I/O, and config loading (no GPU required, finishes in seconds).

## Pipeline Stages

### Stage 1: Generation with Conformal Prediction

| Step | Command | Description | Output |
|------|---------|-------------|--------|
| 1a | `cfa sft` | Fine-tune Llama-2-7B on summarization data (loss on summary tokens only) | SFT model checkpoint |
| 1b | `cfa generate` | Sample 60 responses per prompt, score unique ones with GPT-4o | Frequency dicts + accuracy scores |
| 1c | `cfa calibrate` | Grid-search CP hyperparameters, calibrate quantile threshold | Prediction sets (JSON) |

**Key algorithm — Nonconformity Score:**
```
score = 10 - (freq/total)*10 + (entropy/2)*weight - similarity_to_top*weight_2
```
Lower score = higher confidence = more likely in prediction set.

### Stage 2: AI Feedback with Uncertainty Weights

| Step | Command | Description | Output |
|------|---------|-------------|--------|
| 2a | `cfa feedback` | Pairwise preference annotation via GPT | DPO pairs (JSONL) |
| 2b | `cfa assign-weights` | Weight pairs by CP prediction set membership | Weighted DPO pairs |

**Weight assignment:**
- In 50% coverage set → weight 0.5
- In 80% coverage set only → weight 0.8
- In both → weight 0.65
- In neither → weight 0.0

### Stage 3: Training and Evaluation

| Step | Command | Description | Output |
|------|---------|-------------|--------|
| 3a | `cfa train` | Weighted DPO training (`loss *= weight`) | RLUF model |
| 3b | `cfa infer` | Generate summaries on test set | Predictions (pkl) |
| 3c | `cfa evaluate` | Score with GPT-4o (Accuracy, Relevance, Completeness, Expression) | Scores (pkl) |

## Output Directory Structure

```
outputs/
├── generation/          # Stage 1b outputs
│   ├── generation_llama2.txt
│   ├── generation_llama2_accuracy.txt
│   ├── generation_test_llama2.txt
│   └── response_dict_llama2.pkl
├── calibration/         # Stage 1c outputs
│   ├── prediction_set_quantile0.2_threshold0.7_llama2.json
│   └── prediction_set_quantile0.5_threshold0.7_llama2.json
├── feedback/            # Stage 2 outputs
│   ├── dpo_data_llama2.json
│   └── dpo_data_llama2_withuncertainty.json
├── inference/           # Stage 3b outputs
│   ├── test_dict_question.pkl
│   └── test_dict_RLUF.pkl
└── evaluation/          # Stage 3c outputs
    └── evaluation_scores_llama2.pkl
```

## Configuration

All hyperparameters are in `configs/default.yaml`. Key settings:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `generation.calibration_size` | 50 | Number of calibration samples |
| `generation.sampling_num` | 60 | Samples per prompt |
| `generation.temperature` | 0.35 | Sampling temperature |
| `conformal.quantile_bars` | [0.2, 0.5] | Coverage levels for CP |
| `conformal.accuracy_threshold` | 0.7 | GPT score threshold for "correct" |
| `dpo.learning_rate` | 1.5e-6 | DPO training learning rate |

## Hardware Requirements

| Stage | GPU Memory | Estimated Time |
|-------|-----------|----------------|
| SFT | ~24GB (bfloat16) | ~2-4 hours |
| Generation | ~16GB (bfloat16) | ~6-12 hours (60 samples x 50+ prompts) |
| Calibration | CPU only | ~10-30 minutes |
| Feedback | CPU + OpenAI API | ~1-2 hours (API dependent) |
| DPO Training | ~24GB (float16) | ~2-4 hours |
| Inference | ~16GB (bfloat16) | ~1-2 hours |
| Evaluation | CPU + OpenAI API | ~1-2 hours (API dependent) |

## Docker

```bash
# Build
docker build -t cfa .

# Run a stage
docker run --gpus all --env-file .env -v $(pwd)/outputs:/app/outputs cfa sft --config configs/default.yaml
```

## Project Structure

```
Conformal-Feedback-Alignment/
├── cfa/                    # Main package (pip install conformal-feedback-alignment)
│   ├── cli.py              # CLI entry point
│   ├── config.py           # Configuration loading
│   ├── stages/             # Pipeline stage implementations
│   │   ├── sft.py          # SFT training
│   │   ├── generation.py   # Multi-sample generation + scoring
│   │   ├── calibration.py  # Conformal prediction
│   │   ├── feedback.py     # AI preference annotation
│   │   ├── weights.py      # Uncertainty weight assignment
│   │   ├── train.py        # Weighted DPO training
│   │   ├── inference.py    # Test set inference
│   │   └── evaluation.py   # GPT-4o evaluation
│   ├── models/
│   │   └── weighted_dpo.py # WeightedDPOTrainer
│   └── utils/
│       ├── text_processing.py
│       ├── scoring.py
│       └── io.py
├── configs/
│   ├── default.yaml        # Default configuration
│   └── mwe.yaml            # Minimal working example config
├── tests/                  # Unit tests (40 tests)
├── requirements.txt
├── pyproject.toml
├── Makefile
├── Dockerfile
└── .env.example
```

## Dependencies

- `torch>=2.3.0`
- `transformers>=4.51.0`
- `trl>=0.7.0`
- `datasets>=3.5.0`
- `openai>=0.28.1`
- `gensim>=4.3.0`
- `accelerate>=0.30.0`

## Citation

If you find this work useful, please cite our paper:

```bibtex
@inproceedings{chen-etal-2026-conformal,
    title = "Conformal Feedback Alignment: Quantifying Answer-Level Reliability for Robust {LLM} Alignment",
    author = "Chen, Tiejin  and
      Liu, Xiaoou  and
      Nandam, Vishnu  and
      Liou, Kuan-Ru  and
      Wei, Hua",
    editor = "Demberg, Vera  and
      Inui, Kentaro  and
      Marquez, Llu{\'i}s",
    booktitle = "Findings of the {A}ssociation for {C}omputational {L}inguistics: {EACL} 2026",
    month = mar,
    year = "2026",
    address = "Rabat, Morocco",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2026.findings-eacl.184/",
    pages = "3561--3572",
    ISBN = "979-8-89176-386-9",
}
```

## License

MIT License. See [LICENSE](LICENSE) for details.
