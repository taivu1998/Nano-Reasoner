import os
import torch
import torch.nn as nn

# Try to import Unsloth, fall back to standard transformers if not available
UNSLOTH_AVAILABLE = False
try:
    from unsloth import FastLanguageModel
    UNSLOTH_AVAILABLE = True
except (ImportError, NotImplementedError) as e:
    print(f"Unsloth not available ({e}), using standard transformers")
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training

# Auto-detect Flash Attention
FLASH_ATTN_AVAILABLE = False
try:
    import flash_attn
    FLASH_ATTN_AVAILABLE = True
    print("✓ Flash Attention 2 available")
except ImportError:
    print("⚠ Flash Attention not installed - using eager attention (slower)")


class UnifiedPolicyModel(nn.Module):
    """
    Wraps Unsloth FastLanguageModel (or falls back to standard transformers)
    and conditionally adds a Value Head (Critic).
    """
    def __init__(self, model_name: str, algo: str, max_seq_length: int = 2048, load_in_4bit: bool = True):
        super().__init__()
        self.algo = algo.upper()
        self.device = None  # Will be set when .to() is called
        self.model_name = model_name
        self.use_unsloth = UNSLOTH_AVAILABLE and torch.cuda.is_available()

        if self.use_unsloth:
            # Use Unsloth for optimized training (requires CUDA)
            self.model, self.tokenizer = FastLanguageModel.from_pretrained(
                model_name,
                load_in_4bit=load_in_4bit,
                max_seq_length=max_seq_length,
                dtype=None
            )

            # Configure tokenizer for generation
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            self.tokenizer.padding_side = "left"

            # Add LoRA Adapters via Unsloth
            self.model = FastLanguageModel.get_peft_model(
                self.model,
                r=16,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                lora_alpha=32,
                lora_dropout=0,
                bias="none",
                use_gradient_checkpointing="unsloth"
            )
        else:
            # Fallback: Standard transformers + PEFT + bitsandbytes
            print(f"Loading model with standard transformers: {model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

            # Configure tokenizer
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            self.tokenizer.padding_side = "left"

            # Auto-select attention implementation
            attn_impl = "flash_attention_2" if FLASH_ATTN_AVAILABLE else "eager"
            print(f"Using attention: {attn_impl}")

            # Load model with optional 4-bit quantization
            if load_in_4bit and torch.cuda.is_available():
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_use_double_quant=True,
                )
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    quantization_config=bnb_config,
                    device_map="auto",
                    trust_remote_code=True,
                    attn_implementation=attn_impl,
                    torch_dtype=torch.bfloat16,
                )
                # Enable gradient checkpointing - required to fit in memory
                self.model = prepare_model_for_kbit_training(self.model, use_gradient_checkpointing=True)
                print("✓ Gradient checkpointing enabled")
            else:
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                    device_map="auto" if torch.cuda.is_available() else None,
                    trust_remote_code=True,
                    attn_implementation=attn_impl if torch.cuda.is_available() else "eager",
                )
                self.model.gradient_checkpointing_enable()

            # Add LoRA via PEFT
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=16,
                lora_alpha=32,
                lora_dropout=0.0,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                bias="none",
            )
            self.model = get_peft_model(self.model, lora_config)
            self.model.print_trainable_parameters()

        # Conditional Critic (Only for PPO) - initialized but moved to device later
        self.critic = None
        if self.algo == 'PPO':
            # Simple scalar head projecting from the last hidden state
            hidden_size = self.model.config.hidden_size
            self.critic = nn.Linear(hidden_size, 1).to(torch.bfloat16 if torch.cuda.is_available() else torch.float32)

    def to(self, device):
        """Override to() to handle critic placement."""
        self.device = device
        if not self.use_unsloth or not torch.cuda.is_available():
            # Only move if not using device_map="auto"
            self.model = self.model.to(device)
        if self.critic is not None:
            self.critic = self.critic.to(device)
        return self

    def forward(self, input_ids, attention_mask=None):
        """Standard forward pass returning logits and optional values.

        Args:
            input_ids: Either a tensor of shape [B, T] or can be passed as first positional arg
            attention_mask: Optional attention mask of shape [B, T]
        """
        # We need hidden states only if using Critic
        output_hidden = (self.critic is not None)

        # Generate attention mask if not provided
        if attention_mask is None:
            attention_mask = (input_ids != self.tokenizer.pad_token_id).long()

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=output_hidden,
            use_cache=False,  # Disable cache during training for memory efficiency
        )
        logits = outputs.logits

        values = None
        if self.critic is not None:
            # Use last hidden state for value prediction
            last_hidden = outputs.hidden_states[-1]
            values = self.critic(last_hidden).squeeze(-1)

        return logits, values

    def generate(self, **kwargs):
        return self.model.generate(**kwargs)

    def save_pretrained(self, path):
        """Save model, tokenizer, and optionally critic to path."""
        os.makedirs(path, exist_ok=True)
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        if self.critic:
            torch.save(self.critic.state_dict(), f"{path}/critic.pt")
        # Save training info
        with open(f"{path}/training_info.txt", "w") as f:
            f.write(f"algorithm: {self.algo}\n")
            f.write(f"base_model: {self.model_name}\n")

    def load_adapter_from_checkpoint(self, path):
        """Load LoRA adapter weights from a checkpoint."""
        from peft import set_peft_model_state_dict
        import safetensors.torch

        adapter_safetensors = os.path.join(path, "adapter_model.safetensors")
        adapter_bin = os.path.join(path, "adapter_model.bin")

        if os.path.exists(adapter_safetensors):
            state_dict = safetensors.torch.load_file(adapter_safetensors)
            set_peft_model_state_dict(self.model, state_dict)
            print(f"✓ Loaded adapter from {path}")
        elif os.path.exists(adapter_bin):
            state_dict = torch.load(adapter_bin, map_location="cuda" if torch.cuda.is_available() else "cpu")
            set_peft_model_state_dict(self.model, state_dict)
            print(f"✓ Loaded adapter from {path}")
        else:
            print(f"⚠ No adapter found at {path}")

        if self.critic and os.path.exists(f"{path}/critic.pt"):
            self.critic.load_state_dict(torch.load(f"{path}/critic.pt", map_location="cuda" if torch.cuda.is_available() else "cpu"))
            print(f"✓ Loaded critic from {path}")
