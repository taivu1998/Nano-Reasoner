import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
import re
import gc

class UnifiedReasoningTrainer:
    def __init__(self, policy_model, config, device):
        self.policy = policy_model
        self.config = config
        self.device = device

        # Support both flat config (notebook style) and nested config (original style)
        if 'training' in config:
            # Nested config format
            self.algo = config['training']['algo'].upper()
            self.G = config['training']['group_size']
            lr = float(config['training']['learning_rate'])
            self.ppo_epochs = config['training'].get('ppo_epochs', 2)
            self.max_new_tokens = config['training'].get('max_new_tokens', 384)
        else:
            # Flat config format (notebook style)
            self.algo = config['algo'].upper()
            self.G = config['group_size']
            lr = float(config['learning_rate'])
            self.ppo_epochs = config.get('ppo_epochs', 2)
            self.max_new_tokens = config.get('max_new_tokens', 384)

        # Optimizers
        self.optimizer = AdamW(self.policy.model.parameters(), lr=lr)
        self.critic_optimizer = None
        if self.policy.critic:
            self.critic_optimizer = AdamW(self.policy.critic.parameters(), lr=1e-4)

    # ================= UTILS =================

    def extract_answer(self, text):
        """Extract answer from \\boxed{...}, handling nested braces."""
        # Find \boxed{ and then match balanced braces
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

    def compute_rewards(self, completions, ground_truths):
        """Binary reward: 1.0 if correct, 0.0 otherwise."""
        rewards = []
        for comp, gt in zip(completions, ground_truths):
            pred = self.extract_answer(comp)
            gt_str = str(gt).strip()
            # Compare extracted answer with ground truth (normalize whitespace)
            pred_normalized = pred.strip().replace(" ", "")
            gt_normalized = gt_str.replace(" ", "")
            r = 1.0 if pred_normalized == gt_normalized else 0.0
            rewards.append(r)
        return torch.tensor(rewards, device=self.device, dtype=torch.float32)

    def compute_gae(self, rewards, values, gamma=0.99, lam=0.95):
        """Generalized Advantage Estimation for sparse rewards."""
        # rewards: [B, T], values: [B, T]
        advantages = torch.zeros_like(rewards)
        last_gae_lam = 0
        seq_len = rewards.size(1)

        for t in reversed(range(seq_len)):
            next_val = values[:, t + 1] if t + 1 < seq_len else 0.0
            delta = rewards[:, t] + gamma * next_val - values[:, t]
            last_gae_lam = delta + gamma * lam * last_gae_lam
            advantages[:, t] = last_gae_lam

        returns = advantages + values
        return advantages, returns

    def compute_kl_divergence(self, log_probs, old_log_probs, mask):
        """Compute KL divergence between current and old policy.

        KL(old || new) = sum(old_prob * (old_log_prob - new_log_prob))
        Since we have log probs: KL = sum(exp(old_log_prob) * (old_log_prob - new_log_prob))
        Approximation: KL ≈ mean((old_log_prob - new_log_prob)) when policies are close
        """
        # More stable approximation: 0.5 * mean((log_ratio)^2) where log_ratio = new - old
        log_ratio = log_probs - old_log_probs
        kl = 0.5 * ((log_ratio ** 2) * mask).sum() / mask.sum()
        return kl.item()

    def compute_entropy(self, logits, mask):
        """Compute policy entropy for exploration monitoring."""
        # logits: [B, T, V]
        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)
        entropy_per_token = -(probs * log_probs).sum(dim=-1)  # [B, T]
        # Average entropy over non-padded tokens
        entropy = (entropy_per_token * mask).sum() / mask.sum()
        return entropy.item()

    # ================= LOSS FUNCTIONS =================

    def loss_ppo(self, log_probs, old_log_probs, advantages, returns, values, mask):
        # 1. Token-level Clipping with proper masking
        ratio = torch.exp(log_probs - old_log_probs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 0.8, 1.2) * advantages

        # Apply mask and properly average
        policy_loss = -(torch.min(surr1, surr2) * mask).sum() / mask.sum()

        # 2. Value Function Loss (only on non-padded tokens)
        value_diff = (values - returns) ** 2
        value_loss = (value_diff * mask).sum() / mask.sum()

        return policy_loss + 0.5 * value_loss

    def loss_grpo(self, log_probs, old_log_probs, advantages, mask):
        # Advantages are already calculated and expanded
        ratio = torch.exp(log_probs - old_log_probs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 0.8, 1.2) * advantages

        loss = -torch.min(surr1, surr2) * mask
        return loss.sum() / mask.sum()

    def loss_dr_grpo(self, log_probs, old_log_probs, advantages, mask):
        # Length Correction Factor (L_i / mu_L)
        B = advantages.shape[0] // self.G
        lengths = mask.sum(dim=1).float() # [B*G]
        lengths_grouped = lengths.view(B, self.G)
        avg_len = lengths_grouped.mean(dim=1, keepdim=True)

        # Expand back to [B*G, 1]
        length_scale = (lengths_grouped / (avg_len + 1e-6)).view(-1, 1).expand_as(log_probs)

        ratio = torch.exp(log_probs - old_log_probs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 0.8, 1.2) * advantages

        loss = -torch.min(surr1, surr2) * length_scale
        return (loss * mask).sum() / mask.sum()

    def loss_gspo(self, log_probs, old_log_probs, advantages, mask):
        # Geometric Mean Ratio
        log_diff = (log_probs - old_log_probs) * mask
        sum_log_diff = log_diff.sum(dim=1)
        seq_len = mask.sum(dim=1)

        rho_seq = torch.exp(sum_log_diff / (seq_len + 1e-6)).unsqueeze(-1)

        surr1 = rho_seq * advantages
        surr2 = torch.clamp(rho_seq, 0.8, 1.2) * advantages

        loss = -torch.min(surr1, surr2)
        return loss.mean()

    def loss_dapo(self, log_probs, old_log_probs, rewards_grouped, advantages, mask):
        # Dynamic Sampling: Filter groups with 0 variance
        valid_mask = rewards_grouped.std(dim=1) > 0 # [B]

        # Decoupled Clipping
        ratio = torch.exp(log_probs - old_log_probs)
        upper = torch.where(advantages > 0, 1.28, 1.20)
        lower = 0.8

        surr2 = torch.clamp(ratio, lower, upper) * advantages
        loss = -torch.min(ratio * advantages, surr2) * mask

        # Apply valid mask
        valid_expand = valid_mask.repeat_interleave(self.G).view(-1, 1).expand_as(loss)
        if valid_expand.sum() == 0: return torch.tensor(0.0, device=self.device, requires_grad=True)

        return (loss * valid_expand).sum() / (mask * valid_expand).sum()

    def loss_grpo_lead(self, log_probs, old_log_probs, rewards, mask):
        # Recalculate advantages with Difficulty & Length penalty
        lengths = mask.sum(dim=1).float()
        correct = (rewards == 1.0)

        if correct.any():
            # Length Penalty (Z-score)
            z_score = (lengths - lengths[correct].mean()) / (lengths[correct].std() + 1e-6)
            rewards = torch.where(correct, rewards * torch.exp(-0.1 * z_score.abs()), rewards)

        # Difficulty Weight
        B = rewards.shape[0] // self.G
        rewards_grouped = rewards.view(B, self.G)
        pass_rate = rewards_grouped.mean(dim=1).repeat_interleave(self.G)
        diff_weight = (2.0 - pass_rate).view(-1, 1).expand_as(log_probs)

        # Standard GRPO Norm
        adv = (rewards_grouped - rewards_grouped.mean(1, keepdim=True)) / (rewards_grouped.std(1, keepdim=True)+1e-6)
        adv = adv.view(-1, 1).expand_as(log_probs) * diff_weight

        ratio = torch.exp(log_probs - old_log_probs)
        loss = -torch.min(ratio*adv, torch.clamp(ratio, 0.8, 1.2)*adv) * mask
        return loss.sum() / mask.sum()

    # ================= MAIN LOOP =================

    def train_step(self, batch):
        # --- PATH 1: SFT (Cold Start) ---
        if self.algo == 'SFT':
            texts = batch  # List of strings
            inputs = self.policy.tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=2048).to(self.device)
            outputs = self.policy.model(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask, use_cache=False)

            # Shift for CausalLM (use .clone() to avoid in-place issues)
            logits = outputs.logits[..., :-1, :].clone()
            labels = inputs.input_ids[..., 1:].clone()
            attention_mask = inputs.attention_mask[..., 1:]

            # Compute loss only on non-padded tokens (use -100 to ignore padding)
            labels = labels.masked_fill(attention_mask == 0, -100)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1), ignore_index=-100)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Return metrics dict for consistency
            metrics = {
                'loss': loss.item(),
                'reward': 0.0,
                'kl_divergence': 0.0,
                'entropy': 0.0,
                'avg_response_length': 0.0,
                'accuracy': 0.0,
            }
            return metrics

        # --- PATH 2: RL (Inner Optimization Loop) ---
        prompts = batch['prompts']
        ground_truths = batch['ground_truths']

        # 1. Rollout (Repeat Prompts G times)
        prompts_expanded = [p for p in prompts for _ in range(self.G)]
        gt_expanded = [gt for gt in ground_truths for _ in range(self.G)]

        inputs = self.policy.tokenizer(prompts_expanded, return_tensors="pt", padding=True, padding_side="left").to(self.device)

        # Generation (no grad)
        with torch.no_grad():
            self.policy.model.eval()
            outputs = self.policy.generate(
                **inputs, max_new_tokens=self.max_new_tokens, do_sample=True, temperature=0.8, use_cache=True
            )

        # 2. Process Rollout Data
        prompt_len = inputs.input_ids.shape[1]
        completion_ids = outputs[:, prompt_len:]
        attention_mask = (completion_ids != self.policy.tokenizer.pad_token_id).float()

        decoded = self.policy.tokenizer.batch_decode(completion_ids, skip_special_tokens=True)
        rewards = self.compute_rewards(decoded, gt_expanded) # [B*G]

        # Free generation cache
        del inputs
        torch.cuda.empty_cache()

        # 3. Compute OLD Policy Log Probs & Values (Reference)
        with torch.no_grad():
            logits, old_values = self.policy(outputs)
            logits = logits[:, prompt_len-1:-1, :].clone()

            # Old Log Probs
            old_log_probs = -F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                completion_ids.reshape(-1),
                reduction='none'
            ).view(completion_ids.shape)

            if old_values is not None:
                old_values = old_values[:, prompt_len-1:-1].clone()

            # Free memory
            del logits
            torch.cuda.empty_cache()

        # 4. Pre-compute Advantages (Fixed Target for Inner Loop)
        advantages = None
        returns = None
        B = rewards.shape[0] // self.G  # Always compute B for use in loss functions

        if self.algo == 'PPO':
            # Sparse reward mapping to last token
            seq_rewards = torch.zeros_like(old_log_probs)
            last_indices = attention_mask.sum(dim=1).long() - 1
            for i, idx in enumerate(last_indices):
                if idx >= 0:
                    seq_rewards[i, idx] = rewards[i]

            # GAE
            advantages, returns = self.compute_gae(seq_rewards, old_values)
            # Normalize advantages
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        else:
            # GRPO Family: Group-based normalization
            rewards_grouped = rewards.view(B, self.G)
            mean_r = rewards_grouped.mean(dim=1, keepdim=True)
            std_r = rewards_grouped.std(dim=1, keepdim=True) + 1e-6
            advantages = (rewards_grouped - mean_r) / std_r
            # Expand to [B*G, SeqLen]
            advantages = advantages.view(-1, 1).expand_as(old_log_probs)

        # 5. Inner Optimization Loop (Multiple Epochs)
        self.policy.model.train()

        for _ in range(self.ppo_epochs):
            # Forward pass (With Grad)
            logits, values = self.policy(outputs)
            logits = logits[:, prompt_len-1:-1, :].clone() # Shift

            if values is not None:
                values = values[:, prompt_len-1:-1]

            log_probs = -F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                completion_ids.reshape(-1),
                reduction='none'
            ).view(completion_ids.shape)

            # The Switch
            if self.algo == 'PPO':
                loss = self.loss_ppo(log_probs, old_log_probs, advantages, returns, values, attention_mask)
            elif self.algo == 'GRPO':
                loss = self.loss_grpo(log_probs, old_log_probs, advantages, attention_mask)
            elif self.algo == 'DR.GRPO':
                loss = self.loss_dr_grpo(log_probs, old_log_probs, advantages, attention_mask)
            elif self.algo == 'GSPO':
                loss = self.loss_gspo(log_probs, old_log_probs, advantages, attention_mask)
            elif self.algo == 'DAPO':
                loss = self.loss_dapo(log_probs, old_log_probs, rewards.view(B, self.G), advantages, attention_mask)
            elif self.algo == 'GRPO-LEAD':
                # Note: LEAD modifies advantages inside based on current batch stats,
                # but we pass raw rewards to recalculate if needed.
                loss = self.loss_grpo_lead(log_probs, old_log_probs, rewards, attention_mask)
            else:
                raise ValueError(f"Unknown algorithm: {self.algo}")

            self.optimizer.zero_grad()
            if self.critic_optimizer: self.critic_optimizer.zero_grad()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.model.parameters(), 1.0)

            self.optimizer.step()
            if self.critic_optimizer: self.critic_optimizer.step()

            # Free intermediate tensors
            del logits
            torch.cuda.empty_cache()

        # Compute metrics after final inner loop iteration
        kl_div = self.compute_kl_divergence(log_probs, old_log_probs, attention_mask)
        avg_length = attention_mask.sum(dim=1).float().mean().item()

        metrics = {
            'loss': loss.item(),
            'reward': rewards.mean().item(),
            'kl_divergence': kl_div,
            'entropy': 0.0,  # Skip entropy computation to save memory
            'avg_response_length': avg_length,
            'accuracy': (rewards == 1.0).float().mean().item(),
        }

        # Cleanup
        del outputs, completion_ids, old_log_probs, log_probs, advantages, attention_mask, rewards
        gc.collect()
        torch.cuda.empty_cache()

        return metrics
