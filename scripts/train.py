import sys
import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.config_parser import load_config
from src.utils import seed_everything, get_logger
from src.model import UnifiedPolicyModel
from src.dataset import MathReasoningDataset, collate_fn
from src.trainer import UnifiedReasoningTrainer


def main():
    config = load_config()
    seed_everything(config.get('seed', 42))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create output directory
    output_dir = config['paths']['output_dir']
    os.makedirs(output_dir, exist_ok=True)

    logger = get_logger("Trainer", output_dir)
    logger.info(f"Algo: {config['training']['algo']} | Device: {device}")

    # 1. Model
    policy = UnifiedPolicyModel(
        config['model']['name'],
        config['training']['algo']
    ).to(device)

    # 2. Data
    mode = 'sft' if config['training']['algo'].upper() == 'SFT' else 'rl'
    dataset_name = config.get('data', {}).get('dataset_name', 'openai/gsm8k')
    dataset = MathReasoningDataset(
        policy.tokenizer,
        max_samples=config['training'].get('max_samples'),
        mode=mode,
        dataset_name=dataset_name
    )
    loader = DataLoader(
        dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        collate_fn=collate_fn
    )

    # 3. Trainer
    trainer = UnifiedReasoningTrainer(policy, config, device)

    # 4. Optional: Initialize wandb
    use_wandb = config.get('logging', {}).get('use_wandb', False)
    if use_wandb:
        try:
            import wandb
            wandb.init(project=config.get('logging', {}).get('project', 'UnifiedRL'), config=config)
        except ImportError:
            logger.warning("wandb not installed, skipping wandb logging")
            use_wandb = False

    # 5. Training Loop
    step = 0
    save_steps = config['training'].get('save_steps', 100)

    for epoch in range(config['training']['epochs']):
        logger.info(f"Starting epoch {epoch + 1}/{config['training']['epochs']}")
        pbar = tqdm(loader, desc=f"Epoch {epoch + 1}")

        for batch in pbar:
            metrics = trainer.train_step(batch)

            step += 1

            # Logging
            if use_wandb:
                wandb.log({**metrics, "step": step, "epoch": epoch + 1})

            # Update progress bar
            pbar.set_description(
                f"E{epoch+1} | L: {metrics['loss']:.4f} | R: {metrics['reward']:.2f} | "
                f"KL: {metrics['kl_divergence']:.4f} | Len: {metrics['avg_response_length']:.0f}"
            )

            # Save checkpoint
            if step % save_steps == 0:
                checkpoint_path = f"{output_dir}/step_{step}"
                policy.save_pretrained(checkpoint_path)
                logger.info(f"Saved checkpoint to {checkpoint_path}")

    # Save final model
    final_path = f"{output_dir}/final"
    policy.save_pretrained(final_path)
    logger.info(f"Training complete. Final model saved to {final_path}")

    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()