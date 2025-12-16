"""Inference script for evaluating trained models."""

import sys
import os
import argparse
import torch
from tqdm import tqdm
import re

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.model import UnifiedPolicyModel
from src.dataset import MathReasoningDataset
from src.utils import seed_everything


def extract_answer(text):
    """Extract answer from \\boxed{...}, handling nested braces."""
    start_pattern = r'\\boxed\{'
    match = re.search(start_pattern, text)
    if not match:
        return ""

    start_idx = match.end()
    brace_count = 1
    idx = start_idx

    while idx < len(text) and brace_count > 0:
        if text[idx] == '{':
            brace_count += 1
        elif text[idx] == '}':
            brace_count -= 1
        idx += 1

    if brace_count == 0:
        return text[start_idx:idx-1]
    return ""


def evaluate(model, tokenizer, dataset, device, num_samples=None, temperature=0.0):
    """Evaluate model on dataset and return accuracy."""
    model.eval()
    correct = 0
    total = 0

    if num_samples:
        num_samples = min(num_samples, len(dataset))
    else:
        num_samples = len(dataset)

    with torch.no_grad():
        for idx in tqdm(range(num_samples), desc="Evaluating"):
            item = dataset[idx]
            prompt = item['prompt']
            ground_truth = str(item['ground_truth']).strip().replace(" ", "")

            # Tokenize and generate
            inputs = tokenizer(prompt, return_tensors="pt", padding=True).to(device)

            if temperature == 0.0:
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=512,
                    do_sample=False
                )
            else:
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=512,
                    do_sample=True,
                    temperature=temperature
                )

            # Decode and extract answer
            generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
            predicted = extract_answer(generated).strip().replace(" ", "")

            if predicted == ground_truth:
                correct += 1
            total += 1

    accuracy = correct / total if total > 0 else 0.0
    return accuracy, correct, total


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained model")
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--dataset', type=str, default='openai/gsm8k', help='Dataset to evaluate on')
    parser.add_argument('--split', type=str, default='test', help='Dataset split')
    parser.add_argument('--num_samples', type=int, default=None, help='Number of samples to evaluate')
    parser.add_argument('--temperature', type=float, default=0.0, help='Sampling temperature (0 for greedy)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()

    seed_everything(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load model
    print(f"Loading model from {args.checkpoint}...")
    # Note: For inference, we don't need the algorithm-specific features
    model = UnifiedPolicyModel(args.checkpoint, algo="GRPO").to(device)

    # Load dataset
    print(f"Loading dataset {args.dataset}...")
    dataset = MathReasoningDataset(
        model.tokenizer,
        split=args.split,
        max_samples=args.num_samples,
        mode='rl',
        dataset_name=args.dataset
    )

    # Evaluate
    print("Starting evaluation...")
    accuracy, correct, total = evaluate(
        model,
        model.tokenizer,
        dataset,
        device,
        num_samples=args.num_samples,
        temperature=args.temperature
    )

    print(f"\nResults:")
    print(f"  Accuracy: {accuracy:.4f} ({correct}/{total})")
    print(f"  Pass@1: {accuracy * 100:.2f}%")


if __name__ == "__main__":
    main()
