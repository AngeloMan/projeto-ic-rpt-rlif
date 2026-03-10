import os
import torch
import re
import sympy
from sympy.parsing.sympy_parser import parse_expr

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
OUTPUT_DIR = "qwen-rlvr-math-v1"

# --- AVALIADOR MATEMÁTICO ROBUSTO ---
def extract_boxed_answer(text):
    match = re.search(r"\\boxed\{([^}]*)\}", text)
    return match.group(1).strip() if match else None

def is_math_equivalent(pred, truth):
    if pred == truth: return True
    # Limpa vírgulas de milhar (ex: 1,000 -> 1000)
    pred_clean = pred.replace(",", "").strip()
    truth_clean = truth.replace(",", "").strip()
    
    # Tenta resolver como float (Cobre 90% do GSM8K)
    try:
        if abs(float(pred_clean) - float(truth_clean)) < 1e-6:
            return True
    except: pass
    
    # Tenta resolver via SymPy (Álgebra)
    try:
        expr_pred = parse_expr(pred_clean)
        expr_truth = parse_expr(truth_clean)
        if sympy.simplify(expr_pred - expr_truth) == 0:
            return True
    except: pass
    
    return False

def reward_rlvr_math(completions, ground_truth, **kwargs):
    rewards = []
    for completion, truth in zip(completions, ground_truth):
        pred = extract_boxed_answer(completion)
        if pred is None:
            rewards.append(-1.0) # Punição de formatação
        elif is_math_equivalent(pred, truth):
            rewards.append(1.0)  # Vitória Real (Supervisionada)
        else:
            rewards.append(-1.0) # Errou a conta
    return rewards

# --- FORMATO GSM8K ---
def format_data_func(example):
    truth_match = example['answer'].split('####')
    ground_truth = truth_match[1].strip() if len(truth_match) > 1 else ""
    
    system_msg = "You are a mathematical reasoning assistant. Think step by step to solve the problem. You must put your final numerical answer inside \\boxed{}."
    prompt_text = (
        f"<|im_start|>system\n{system_msg}<|im_end|>\n"
        f"<|im_start|>user\n{example['question']}\n<|im_end|>\n"
        f"<|im_start|>assistant\n" 
    )
    return {"prompt": prompt_text, "ground_truth": ground_truth}

def main():
    print(f"--- TREINO {OUTPUT_DIR} (RLVR Matemática - SymPy) ---")
    peft_config = LoraConfig(r=16, target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"], task_type=TaskType.CAUSAL_LM, lora_alpha=32, lora_dropout=0.05)
    training_args = GRPOConfig(output_dir=OUTPUT_DIR, learning_rate=1e-6, num_train_epochs=1, per_device_train_batch_size=1, gradient_accumulation_steps=8, num_generations=4, max_prompt_length=600, max_completion_length=512, logging_steps=5, save_steps=50, fp16=True, beta=0.01, use_vllm=False, report_to="none", temperature=0.7)
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    tokenizer.padding_side = "right" 
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token 

    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True)
    model.config.pad_token_id = tokenizer.pad_token_id
    model.resize_token_embeddings(len(tokenizer))

    dataset = load_dataset("gsm8k", "main", split="train").map(format_data_func, batched=False)

    trainer = GRPOTrainer(model=model, reward_funcs=reward_rlvr_math, args=training_args, train_dataset=dataset, peft_config=peft_config, processing_class=tokenizer)
    trainer.train()
    trainer.save_model(OUTPUT_DIR)

if __name__ == "__main__":
    main()