import yaml
import argparse
from typing import Dict, Any


def load_config() -> Dict[str, Any]:
    """
    Loads YAML config and overrides with Command Line Arguments.
    Priority: CLI > YAML > Defaults
    """
    parser = argparse.ArgumentParser(description="Unified RL Framework Trainer")
    parser.add_argument('--config', type=str, required=True, help='Path to YAML config')

    # CLI Overrides
    parser.add_argument('--algo', type=str, help='Algorithm: SFT, PPO, GRPO, DR.GRPO, GSPO, DAPO, GRPO-LEAD')
    parser.add_argument('--group_size', type=int, help='Generations per prompt (G)')
    parser.add_argument('--batch_size', type=int, help='Prompt batch size (B)')
    parser.add_argument('--output_dir', type=str, help='Checkpoint directory')
    parser.add_argument('--epochs', type=int, help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, help='Learning rate')
    parser.add_argument('--max_samples', type=int, help='Max training samples')
    parser.add_argument('--ppo_epochs', type=int, help='Inner loop optimization epochs')
    parser.add_argument('--wandb', action='store_true', help='Enable wandb logging')

    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Ensure nested dicts exist
    if 'training' not in config:
        config['training'] = {}
    if 'paths' not in config:
        config['paths'] = {}
    if 'logging' not in config:
        config['logging'] = {}

    # Apply Overrides
    if args.algo:
        config['training']['algo'] = args.algo
    if args.group_size:
        config['training']['group_size'] = args.group_size
    if args.batch_size:
        config['training']['batch_size'] = args.batch_size
    if args.output_dir:
        config['paths']['output_dir'] = args.output_dir
    if args.epochs:
        config['training']['epochs'] = args.epochs
    if args.learning_rate:
        config['training']['learning_rate'] = args.learning_rate
    if args.max_samples:
        config['training']['max_samples'] = args.max_samples
    if args.ppo_epochs:
        config['training']['ppo_epochs'] = args.ppo_epochs
    if args.wandb:
        config['logging']['use_wandb'] = True

    return config