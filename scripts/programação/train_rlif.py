import os
import sys
import torch
import torch.nn.functional as F
import re

# --- 1. VACINAS DE ESTABILIDADE (CRÍTICO PARA RTX 3060 + WINDOWS) ---
torch.cuda.is_bf16_supported = lambda: False
os.environ["ACCELERATE_MIXED_PRECISION"] = "fp16"

from transformers import modeling_utils
_original_to = modeling_utils.PreTrainedModel.to
def _patched_to(self, *args, **kwargs):
    try:
        return _original_to(self, *args, **kwargs)
    except ValueError as e:
        raise e
modeling_utils.PreTrainedModel.to = _patched_to
# ---------------------------------------------------------------

from datasets import load_dataset
from trl import GRPOTrainer, GRPOConfig
from peft import LoraConfig, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer

# --- CONFIGURAÇÕES ---
MODEL_NAME = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
OUTPUT_DIR = "qwen-rlif-v1"

# --- 2. FORMATAÇÃO DE DADOS ---
def format_data_func(example):
    gold_code = example.get('code', '')
    match = re.search(r"def\s+\w+\s*\(.*?\):", gold_code, re.DOTALL)
    signature = match.group(0) if match else "def solution():"
    
    prompt_raw = example.get('prompt', example.get('text', ''))
    
    system_msg = "You are a Python coding engine. Respond ONLY with the indented function body."
    
    prompt_text = (
        f"<|im_start|>system\n{system_msg}<|im_end|>\n"
        f"<|im_start|>user\nImplement:\n{signature}\n\"\"\"\n{prompt_raw}\n\"\"\"\n<|im_end|>\n"
        f"<|im_start|>assistant\n" 
    )
    
    return {
        "prompt": prompt_text,
        "completion": gold_code 
    }

# --- 3. EXECUÇÃO PRINCIPAL ---
def main():
    print(f"--- INICIANDO TREINO {OUTPUT_DIR} (Self-Certainty | Full Debug) ---")
    
    peft_config = LoraConfig(
        r=16, 
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"],
        task_type=TaskType.CAUSAL_LM,
        lora_alpha=32,
        lora_dropout=0.05,
    )

    training_args = GRPOConfig(
        output_dir=OUTPUT_DIR,
        learning_rate=1e-6, 
        num_train_epochs=1,
        per_device_train_batch_size=1, 
        gradient_accumulation_steps=8, 
        num_generations=4,             
        max_prompt_length=600,
        max_completion_length=512,
        logging_steps=5,
        save_steps=50,
        fp16=True,       
        beta=0.01,       
        use_vllm=False,                
        report_to="none",
        temperature=0.7, 
    )

    print("Carregando Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    tokenizer.padding_side = "right" 
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token 

    print("Carregando Modelo (FP16)...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16, 
        device_map="auto",
        trust_remote_code=True,
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    model.resize_token_embeddings(len(tokenizer))

    # --- 4. FUNÇÃO DE RECOMPENSA (RLIF) ---
    def reward_rlif_certainty(completions, prompts, **kwargs):
        rewards = []
        full_texts = [p + c for p, c in zip(prompts, completions)]
        
        inputs = tokenizer(full_texts, return_tensors="pt", padding=True, padding_side="right").to(model.device)
        
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits 
        
        for i in range(len(full_texts)):
            prompt_inputs = tokenizer(prompts[i], add_special_tokens=False, return_tensors="pt")
            prompt_len = prompt_inputs.input_ids.shape[1]
            
            completion_logits = logits[i, prompt_len-1 : -1, :]
            input_ids = inputs.input_ids[i, prompt_len:]
            attention_mask = inputs.attention_mask[i, prompt_len:]
            
            valid_length = attention_mask.sum().item()
            if valid_length == 0:
                rewards.append(0.0)
                continue
                
            valid_logits = completion_logits[:valid_length]
            valid_ids = input_ids[:valid_length]
            
            token_log_probs = -F.cross_entropy(valid_logits, valid_ids, reduction='none')
            
            certainty_score = token_log_probs.mean().item() + 2.0
            rewards.append(certainty_score)
            
            # --- MONITOR COMPLETO ---
            if i == 0:
                print(f"\nMONITOR RLIF | Confiança: {certainty_score:.4f}")
                print("-" * 50)
                # AGORA MOSTRA TUDO:
                print(completions[i].strip()) 
                print("-" * 50 + "\n")

        return rewards

    print("Carregando Dataset...")
    dataset = load_dataset("mbpp", "sanitized", split="train")
    dataset = dataset.map(format_data_func, batched=False)

    print("Iniciando Trainer...")
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_rlif_certainty, 
        args=training_args,
        train_dataset=dataset,
        peft_config=peft_config,
        processing_class=tokenizer,
    )

    print("TREINO RLIF INICIADO!")
    trainer.train()
    
    print("Salvando modelo final...")
    trainer.save_model(OUTPUT_DIR)
    print(f"Modelo RLIF salvo em: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()