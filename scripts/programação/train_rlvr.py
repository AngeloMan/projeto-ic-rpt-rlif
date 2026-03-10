import os
import torch
import re
import threading
import textwrap

# --- 1. VACINAS DE ESTABILIDADE (RTX 3060 + WINDOWS) ---
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
OUTPUT_DIR = "qwen-rlvr-v1" # Pasta onde o modelo treinado será salvo

# --- 2. FUNÇÃO DE RECOMPENSA EXTERNA (RLVR) ---

def run_code_safe(code_string, timeout=1.0):
    """
    Executa o código com um limite de tempo. 
    Vital no RLVR para o treino não congelar se o modelo gerar um loop infinito (while True).
    """
    result = [False]
    
    def target():
        try:
            # Executa o código num ambiente isolado (dicionário vazio)
            exec(code_string, {})
            result[0] = True
        except Exception:
            pass # Erro de sintaxe, import inválido ou falha no assert
            
    thread = threading.Thread(target=target)
    thread.start()
    thread.join(timeout)
    
    if thread.is_alive():
        return False # Estourou o tempo limite (Timeout)
    return result[0]

def reward_rlvr_execution(completions, test_list, **kwargs):
    """
    A essência do RLVR: A recompensa vem do ambiente (Testes Unitários), não da probabilidade.
    """
    rewards = []
    
    for completion, tests in zip(completions, test_list):
        # 1. Limpa a saída do modelo (remove Markdown)
        code = completion.strip()
        if "```python" in code:
            code = code.split("```python")[1].split("```")[0]
        elif "```" in code:
            code = code.split("```")[1].split("```")[0]
        
        # 2. Prepara o código com os testes do MBPP (ex: assert add(1,2) == 3)
        code = textwrap.dedent(code)
        tests_str = "\n".join(tests)
        full_execution_code = f"{code}\n\n{tests_str}"
        
        # 3. Executa e Recompensa
        passou = run_code_safe(full_execution_code)
        
        if passou:
            rewards.append(1.0)  # Recompensa positiva (Feedback Externo de Sucesso)
        else:
            rewards.append(-1.0) # Punição (Erro de lógica ou compilação)
            
    return rewards

# --- 3. PREPARAÇÃO DOS DADOS ---
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
        "test_list": example['test_list'] # Passamos a lista de testes para o GRPOTrainer usar na recompensa
    }

# --- 4. EXECUÇÃO PRINCIPAL ---
def main():
    print(f"--- INICIANDO TREINO {OUTPUT_DIR} (RLVR - Verificador Externo) ---")
    
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

    print("Carregando Dataset (MBPP)...")
    dataset = load_dataset("mbpp", "sanitized", split="train")
    dataset = dataset.map(format_data_func, batched=False)

    print("Iniciando Trainer...")
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_rlvr_execution, # Usando a recompensa por execução de testes
        args=training_args,
        train_dataset=dataset,
        peft_config=peft_config,
        processing_class=tokenizer,
    )

    print("TREINO RLVR INICIADO! (Modo Execução com Verificador)")
    trainer.train()
    
    print("Salvando modelo final...")
    trainer.save_model(OUTPUT_DIR)
    print(f"Modelo RLVR salvo com sucesso na pasta: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()