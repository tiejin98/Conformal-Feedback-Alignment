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

### 1. Configure

```bash
# Set up your API keys
cp .env.example .env
# Edit .env:
#   OPENAI_API_KEY=sk-...
#   HF_TOKEN=hf_...

# Customize the config (or use the default)
cp configs/default.yaml configs/my_config.yaml
# Edit configs/my_config.yaml with your model paths
```

### 2. Run

```bash
# Run individual stages:
cfa sft --config configs/my_config.yaml
cfa generate --config configs/my_config.yaml
cfa calibrate --config configs/my_config.yaml --quantile 0.2
cfa calibrate --config configs/my_config.yaml --quantile 0.5
cfa feedback --config configs/my_config.yaml
cfa assign-weights --config configs/my_config.yaml
cfa train --config configs/my_config.yaml
cfa infer --config configs/my_config.yaml
cfa evaluate --config configs/my_config.yaml

# Or run the full pipeline:
cfa run-all --config configs/my_config.yaml
```

### 3. Run Tests

```bash
pytest tests/ -v
```

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
| 2a | `cfa feedback` | Pairwise preference annotation via AlpacaFarm | DPO pairs (JSONL) |
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
- `alpaca-farm`
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
