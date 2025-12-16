.PHONY: install train-sft train-grpo train-ppo train-dr-grpo train-gspo train-dapo train-grpo-lead evaluate clean help

help:
	@echo "Unified Reasoning RL Framework"
	@echo ""
	@echo "Setup:"
	@echo "  make install      - Install dependencies"
	@echo ""
	@echo "Training (by algorithm):"
	@echo "  make train-sft       - Cold Start SFT (format tuning)"
	@echo "  make train-grpo      - GRPO (baseline, low VRAM)"
	@echo "  make train-ppo       - PPO (requires critic, high VRAM)"
	@echo "  make train-dr-grpo   - Dr.GRPO (length-corrected)"
	@echo "  make train-gspo      - GSPO (sequence-level ratio)"
	@echo "  make train-dapo      - DAPO (decoupled clipping)"
	@echo "  make train-grpo-lead - GRPO-LEAD (length & difficulty aware)"
	@echo ""
	@echo "Evaluation:"
	@echo "  make evaluate CKPT=path/to/checkpoint"
	@echo ""
	@echo "Maintenance:"
	@echo "  make clean        - Remove cache and checkpoints"

install:
	pip install -r requirements.txt

# ============ Training Commands ============

# 1. Cold Start (Format Tuning) - Run this first!
train-sft:
	python scripts/train.py --config configs/sft.yaml

# 2. GRPO (Low VRAM, standard baseline)
train-grpo:
	python scripts/train.py --config configs/base.yaml --algo GRPO

# 3. PPO (Requires Critic, Higher VRAM)
train-ppo:
	python scripts/train.py --config configs/base.yaml --algo PPO --group_size 4

# 4. Dr.GRPO (Length-corrected GRPO)
train-dr-grpo:
	python scripts/train.py --config configs/base.yaml --algo DR.GRPO

# 5. GSPO (Sequence-level Policy Optimization)
train-gspo:
	python scripts/train.py --config configs/base.yaml --algo GSPO

# 6. DAPO (Decoupled Clipping + Dynamic Sampling)
train-dapo:
	python scripts/train.py --config configs/base.yaml --algo DAPO

# 7. GRPO-LEAD (Length & Difficulty Aware)
train-grpo-lead:
	python scripts/train.py --config configs/base.yaml --algo GRPO-LEAD

# ============ Evaluation ============

evaluate:
ifndef CKPT
	$(error CKPT is not set. Usage: make evaluate CKPT=checkpoints/step_100)
endif
	python scripts/inference.py --checkpoint $(CKPT) --num_samples 500

# ============ Maintenance ============

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	rm -rf logs/ checkpoints/ wandb/
	rm -f *.pyc