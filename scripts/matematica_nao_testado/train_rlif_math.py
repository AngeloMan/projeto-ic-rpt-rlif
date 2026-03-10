import os
import torch
import torch.nn.functional as F

torch.cuda.is_bf16_supported = lambda: False
os.environ["ACCELERATE_MIXED_PRECISION"] = "fp16"

from transformers import modeling_utils
_original_to = modeling_utils.PreTrainedModel.to
def _patched_to(self, *args, **kwargs):
    try: return _original_to(self, *args, **kwargs)
    except ValueError as e: raise e
modeling_utils.PreTrainedModel.to = _patched_to

from datasets import load_dataset
from trl import GRPOTrainer, GRPOConfig
from peft import LoraConfig, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
OUTPUT_DIR = "qwen-rlif-math-v1"

def format_data_func(example):
    system_msg = "You are a mathematical reasoning assistant. Think step by step to solve the problem. You must put your final numerical answer inside \\boxed{}."
    prompt_text = (
        f"<|im_start|>system\n{system_msg}<|im_end|>\n"
        f"<|im_start|>user\n{example['question']}\n<|im_end|>\n"
        f"<|im_start|>assistant\n" 
    )
    return {"prompt": prompt_text}

def main():
    print(f"--- TREINO {OUTPUT_DIR} (RLIF Matemática - Bug Fix) ---")
    peft_config = LoraConfig(r=16, target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"], task_type=TaskType.CAUSAL_LM, lora_alpha=32, lora_dropout=0.05)
    training_args = GRPOConfig(output_dir=OUTPUT_DIR, learning_rate=1e-6, num_train_epochs=1, per_device_train_batch_size=1, gradient_accumulation_steps=8, num_generations=4, max_prompt_length=600, max_completion_length=512, logging_steps=5, save_steps=50, fp16=True, beta=0.01, use_vllm=False, report_to="none", temperature=0.7)
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    tokenizer.padding_side = "right" 
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token 

    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True)
    model.config.pad_token_id = tokenizer.pad_token_id
    model.resize_token_embeddings(len(tokenizer))

    # --- FUNÇÃO RLIF EXATA ---
    def reward_rlif_certainty(completions, prompts, **kwargs):
        rewards = []
        with torch.no_grad():
            for p, c in zip(prompts, completions):
                # Tokeniza blindando as fronteiras
                p_ids = tokenizer(p, return_tensors="pt", add_special_tokens=False).input_ids.to(model.device)
                c_ids = tokenizer(c, return_tensors="pt", add_special_tokens=False).input_ids.to(model.device)
                
                if c_ids.shape[1] == 0:
                    rewards.append(0.0)
                    continue

                input_ids = torch.cat([p_ids, c_ids], dim=1)
                logits = model(input_ids).logits
                prompt_len = p_ids.shape[1]
                
                completion_logits = logits[0, prompt_len-1 : -1, :]
                completion_targets = c_ids[0, :]
                
                token_log_probs = -F.cross_entropy(completion_logits, completion_targets, reduction='none')
                certainty_score = token_log_probs.mean().item() + 2.0
                rewards.append(certainty_score)
        return rewards

    dataset = load_dataset("gsm8k", "main", split="train").map(format_data_func, batched=False)

    trainer = GRPOTrainer(model=model, reward_funcs=reward_rlif_certainty, args=training_args, train_dataset=dataset, peft_config=peft_config, processing_class=tokenizer)
    trainer.train()
    trainer.save_model(OUTPUT_DIR)

if __name__ == "__main__":
    main()