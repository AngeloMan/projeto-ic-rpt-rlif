import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from human_eval.data import write_jsonl, read_problems
from tqdm import tqdm
import re
import sys

# --- CONFIGURAÇÕES ---
BASE_MODEL = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
ADAPTER_PATH = "qwen-rlif" # Pasta gerada pelo seu treino atual
OUTPUT_FILE = "samples_rlif.jsonl"
DEVICE = "cuda"

print(f"Carregando Tokenizer: {BASE_MODEL}...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)

print(f"Carregando Modelo Base: {BASE_MODEL}...")
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)

print(f"Ajustando embeddings...")
base_model.resize_token_embeddings(len(tokenizer))

print(f"Acoplando RLIF (LoRA): {ADAPTER_PATH}...")
model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
model.eval()

problems = read_problems()

def limpar_codigo(completion):
    """
    Remove Markdown e garante que só sobre o código Python puro.
    """
    # 1. Remove Markdown ```python
    if "```python" in completion:
        completion = completion.split("```python")[1].split("```")[0]
    elif "```" in completion:
        completion = completion.split("```")[1].split("```")[0]
    
    # 2. Remove espaços extras nas pontas
    completion = completion.strip()
    
    # 3. Remove repetição da assinatura (def func)
    lines = completion.split('\n')
    if len(lines) > 0 and lines[0].strip().startswith("def "):
        completion = "\n".join(lines[1:])
        
    return completion

samples = []
print("Gerando para avaliação...")

# Prompt igual ao treino para ativar a memória do modelo
system_msg = "You are a Python coding engine. Respond ONLY with the indented function body."

for task_id in tqdm(problems):
    problem = problems[task_id]
    
    prompt_text = (
        f"<|im_start|>system\n{system_msg}<|im_end|>\n"
        f"<|im_start|>user\nImplement:\n{problem['prompt']}\n<|im_end|>\n"
        f"<|im_start|>assistant\n" 
    )
    
    inputs = tokenizer(prompt_text, return_tensors="pt").to(DEVICE)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=512, 
            temperature=0.1, 
            top_p=0.95,
            do_sample=True
        )
    
    raw_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    if "assistant" in raw_output:
        raw_completion = raw_output.split("assistant")[-1]
    else:
        raw_completion = raw_output

    # LIMPEZA CRÍTICA AQUI
    final_completion = limpar_codigo(raw_completion)
    
    samples.append(dict(task_id=task_id, completion=final_completion))

write_jsonl(OUTPUT_FILE, samples)
print(f"Arquivo {OUTPUT_FILE} gerado.")
