from datasets import load_dataset
from torch.utils.data import Dataset
import torch


class MathReasoningDataset(Dataset):
    """
    Wrapper for GSM8K/Math datasets.
    Handles prompt formatting for Qwen-Math.
    """
    def __init__(self, tokenizer, split="train", max_samples=None, mode="rl", dataset_name="openai/gsm8k"):
        self.tokenizer = tokenizer
        self.mode = mode  # 'rl' or 'sft'
        self.dataset_name = dataset_name

        # Load dataset - handle different dataset schemas
        if "openr1" in dataset_name.lower():
            self.data = load_dataset(dataset_name, split=split)
            self.problem_key = "problem"
            self.solution_key = "solution"
            self.answer_key = "answer"
        elif "gsm8k" in dataset_name.lower():
            self.data = load_dataset(dataset_name, "main", split=split)
            self.problem_key = "question"
            self.solution_key = "answer"
            self.answer_key = "answer"  # GSM8K has answer embedded in solution
        else:
            # Default fallback
            self.data = load_dataset(dataset_name, split=split)
            self.problem_key = "problem"
            self.solution_key = "solution"
            self.answer_key = "answer"

        if max_samples:
            max_samples = min(max_samples, len(self.data))
            self.data = self.data.select(range(max_samples))

    def __len__(self):
        return len(self.data)

    def _extract_gsm8k_answer(self, solution_text):
        """Extract final numeric answer from GSM8K format (#### answer)."""
        if "####" in solution_text:
            return solution_text.split("####")[-1].strip()
        return solution_text.strip()

    def __getitem__(self, idx):
        item = self.data[idx]

        # Get problem text (handle different field names)
        problem = item.get(self.problem_key, item.get("question", item.get("problem", "")))

        # Standard Qwen-Math Prompt Template
        # We append <think> to force the model into CoT mode immediately for RL
        prompt = (
            "<|im_start|>system\n"
            "Please reason step by step and put your final answer within \\boxed{}.<|im_end|>\n"
            "<|im_start|>user\n"
            f"{problem}<|im_end|>\n"
            "<|im_start|>assistant\n"
            "<think>"
        )

        if self.mode == 'sft':
            # For SFT, we need the full completion with proper format
            solution = item.get(self.solution_key, item.get("solution", ""))
            # Format solution with think tags and boxed answer for SFT
            full_text = prompt + "\n" + solution + "\n</think>\n\\boxed{" + self._extract_gsm8k_answer(solution) + "}<|im_end|>"
            return {"text": full_text}

        # For RL mode, extract ground truth answer
        if "gsm8k" in self.dataset_name.lower():
            # GSM8K: answer is embedded in solution after ####
            answer = self._extract_gsm8k_answer(item.get(self.answer_key, ""))
        else:
            answer = item.get(self.answer_key, item.get("answer", ""))

        return {
            "prompt": prompt,
            "ground_truth": answer
        }

def collate_fn(batch):
    """Custom collate for text-based RL rollouts."""
    if "text" in batch[0]: # SFT Mode
        return [b["text"] for b in batch]
        
    return {
        "prompts": [b["prompt"] for b in batch],
        "ground_truths": [b["ground_truth"] for b in batch]
    }