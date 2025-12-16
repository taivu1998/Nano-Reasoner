# Nano-Reasoner

**A Unified Reinforcement Learning Framework for Mathematical Reasoning**

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.4+](https://img.shields.io/badge/pytorch-2.4+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Train small language models to reason like giants. Nano-Reasoner implements state-of-the-art RL algorithms for mathematical reasoning, achieving **84%+ accuracy on GSM8K** with just a 1.5B parameter model.

---

## Highlights

- **6 RL Algorithms** — PPO, GRPO, Dr.GRPO, GSPO, DAPO, and GRPO-LEAD in one unified framework
- **Memory Efficient** — 4-bit quantization + LoRA + gradient checkpointing (runs on 16GB GPUs)
- **Two-Phase Training** — SFT cold start followed by RL fine-tuning for optimal results
- **Production Ready** — Checkpoint resumption, W&B logging, and modular design
- **Colab Compatible** — Full training pipeline in a single notebook

---

## Training Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                     NANO-REASONER PIPELINE                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Phase 1: SFT Cold Start                                        │
│  ━━━━━━━━━━━━━━━━━━━━━━━                                        │
│  • Teaches model the <think>...</think> format                  │
│  • Train for 1 epoch on format examples                         │
│  • Saves to: checkpoints/sft/                                   │
│                          │                                       │
│                          ▼                                       │
│  Phase 2: RL Training (Choose Algorithm)                        │
│  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━                        │
│  • Loads SFT checkpoint as base                                 │
│  • PPO / GRPO / Dr.GRPO / GSPO / DAPO / GRPO-LEAD              │
│  • Optimizes for correctness via reward signal                  │
│  • Saves to: checkpoints/<algorithm>/                           │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

**Why SFT First?** RL algorithms require the model to spontaneously emit `<think>` tags. An untrained model won't do this. SFT "cold start" teaches the format before RL optimizes for correctness.

---

## Supported Algorithms

| Algorithm | Description | Memory | Best For |
|-----------|-------------|--------|----------|
| **GRPO** | Group Relative Policy Optimization | Low | Baseline, stable training |
| **Dr.GRPO** | Length-corrected GRPO | Low | Avoiding length bias |
| **PPO** | Proximal Policy Optimization | High | When you have VRAM to spare |
| **GSPO** | Geometric Sequence Policy Optimization | Low | Sequence-level optimization |
| **DAPO** | Decoupled Advantage Policy Optimization | Low | Dynamic sampling scenarios |
| **GRPO-LEAD** | Length & Difficulty Aware GRPO | Low | Curriculum-style training |

---

## Installation

### Prerequisites

- Python 3.9+
- CUDA 11.8+ (for GPU training)
- 16GB+ GPU memory recommended

### Quick Install

```bash
# Clone the repository
git clone https://github.com/your-org/nano-reasoner.git
cd nano-reasoner

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
make install
# Or manually: pip install -r requirements.txt
```

### Optional: Flash Attention 2 (2-3x speedup on A100/H100)

```bash
pip install flash-attn --no-build-isolation
```

---

## Quick Start

### 1. SFT Cold Start (Required First Step)

```bash
make train-sft
```

This teaches the model the reasoning format (`<think>...</think>` and `\boxed{}`).

### 2. RL Training

Choose your algorithm:

```bash
# Recommended: Dr.GRPO (length-corrected, stable)
make train-dr-grpo

# Or try other algorithms
make train-grpo       # Baseline GRPO
make train-ppo        # PPO (higher memory)
make train-gspo       # Sequence-level optimization
make train-dapo       # Decoupled clipping
make train-grpo-lead  # Length & difficulty aware
```

### 3. Evaluation

```bash
make evaluate CKPT=checkpoints/dr_grpo/best
```

---

## Configuration

Edit `configs/base.yaml` to customize training:

```yaml
seed: 42

model:
  name: "Qwen/Qwen2.5-Math-1.5B-Instruct"

data:
  dataset_name: "openai/gsm8k"  # or "HuggingFaceH4/openr1-math-220k"

training:
  algo: "DR.GRPO"      # PPO, GRPO, DR.GRPO, GSPO, DAPO, GRPO-LEAD
  group_size: 8        # Number of samples per prompt for GRPO variants
  batch_size: 2        # Batch size per GPU
  epochs: 1
  learning_rate: 5.0e-6
  max_samples: 5000    # null for full dataset
  max_new_tokens: 384  # Max reasoning length
  ppo_epochs: 2        # Inner optimization iterations
  save_steps: 100

paths:
  output_dir: "checkpoints"

logging:
  use_wandb: false
  project: "NanoReasoner"
```

### Memory Guide

| GPU | VRAM | Recommended Settings |
|-----|------|---------------------|
| T4 | 16GB | `batch_size: 1, group_size: 2` |
| L4 | 24GB | `batch_size: 1, group_size: 4` |
| A100 40GB | 40GB | `batch_size: 2, group_size: 8` |
| A100 80GB | 80GB | `batch_size: 8, group_size: 8` |

---

## Results

Training on GSM8K with Qwen2.5-Math-1.5B-Instruct:

| Algorithm | Accuracy | Training Time | Notes |
|-----------|----------|---------------|-------|
| Base Model | ~65% | - | Zero-shot |
| + SFT | ~72% | ~30 min | Format tuning only |
| + Dr.GRPO | **84%** | ~10 hrs | 5000 samples, A100 |
| + GRPO | 82% | ~10 hrs | Baseline RL |

*Results on A100-80GB with default hyperparameters.*

---

## Project Structure

```
nano-reasoner/
├── src/
│   ├── __init__.py          # Package exports
│   ├── config_parser.py     # YAML config loading
│   ├── dataset.py           # MathReasoningDataset
│   ├── model.py             # UnifiedPolicyModel (4-bit + LoRA)
│   ├── trainer.py           # UnifiedReasoningTrainer (all algorithms)
│   └── utils.py             # Utilities (seeding, logging, checkpoints)
├── scripts/
│   ├── train.py             # Training entry point
│   └── inference.py         # Evaluation script
├── configs/
│   ├── base.yaml            # RL training config
│   └── sft.yaml             # SFT cold start config
├── notebooks/
│   └── unified_rl_training_v2.ipynb  # Colab notebook
├── Makefile                 # Training commands
├── requirements.txt         # Dependencies
└── pyproject.toml          # Package metadata
```

---

## Colab Notebook

For cloud training, use the included Jupyter notebook:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](notebooks/unified_rl_training_v2.ipynb)

The notebook includes:
- Google Drive integration for checkpoint persistence
- Interactive configuration widgets
- Real-time training metrics
- Built-in evaluation and inference

---

## Advanced Usage

### Resume Training

Training automatically resumes from the latest checkpoint:

```python
from src import find_latest_checkpoint

checkpoint_dir = "checkpoints/dr_grpo"
path, step = find_latest_checkpoint(checkpoint_dir)
print(f"Resuming from step {step}: {path}")
```

### Custom Datasets

Extend `MathReasoningDataset` for new datasets:

```python
from src import MathReasoningDataset

dataset = MathReasoningDataset(
    tokenizer=tokenizer,
    split="train",
    max_samples=10000,
    mode="rl",  # or "sft"
    dataset_name="your-org/your-math-dataset"
)
```

### Programmatic Training

```python
from src import (
    UnifiedPolicyModel,
    UnifiedReasoningTrainer,
    MathReasoningDataset,
    seed_everything
)

seed_everything(42)

# Initialize model
model = UnifiedPolicyModel("Qwen/Qwen2.5-Math-1.5B-Instruct", algo="DR.GRPO")

# Configure trainer
config = {
    "algo": "DR.GRPO",
    "group_size": 8,
    "learning_rate": 5e-6,
    "max_new_tokens": 384,
    "ppo_epochs": 1
}
trainer = UnifiedReasoningTrainer(model, config, device="cuda")

# Train
for batch in dataloader:
    metrics = trainer.train_step(batch)
    print(f"Loss: {metrics['loss']:.4f}, Accuracy: {metrics['accuracy']:.1%}")
```

---

## Algorithm Details

### GRPO (Group Relative Policy Optimization)

Standard baseline that normalizes advantages within groups of samples for the same prompt.

### Dr.GRPO (Length-Corrected)

Addresses length bias by scaling gradients inversely with response length:

```
scale_i = L_i / mean(L_group)
```

### GRPO-LEAD (Length & Difficulty Aware)

Combines length penalty with difficulty weighting based on pass rate:

```
difficulty_weight = 2.0 - pass_rate
length_penalty = exp(-0.1 * |z_score|)
```

---

## Troubleshooting

### Out of Memory

1. Reduce `batch_size` and `group_size`
2. Enable gradient checkpointing (default: enabled)
3. Use 4-bit quantization (default: enabled)

### Accuracy Stuck at 0%

1. Verify SFT phase completed successfully
2. Check reward function: `trainer.extract_answer()` must find `\boxed{}`
3. Ensure model generates `<think>` tags

### Slow Training

1. Install Flash Attention 2
2. Use larger batch sizes if memory allows
3. Check GPU utilization with `nvidia-smi`

---

## Citation

If you use Nano-Reasoner in your research, please cite:

```bibtex
@software{nano_reasoner,
  title = {Nano-Reasoner: Unified RL for Mathematical Reasoning},
  year = {2024},
  url = {https://github.com/your-org/nano-reasoner}
}
```

---

## License

MIT License - see [LICENSE](LICENSE) for details.

---

## Acknowledgments

- [Qwen2.5-Math](https://github.com/QwenLM/Qwen2.5-Math) for the base model
- [Unsloth](https://github.com/unslothai/unsloth) for optimized training
- [GRPO](https://arxiv.org/abs/2402.03300) and [Dr.GRPO](https://arxiv.org/abs/2503.02062) papers for algorithm inspiration
