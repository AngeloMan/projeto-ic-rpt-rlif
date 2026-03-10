import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from datasets import load_dataset
from tqdm import tqdm
import re
import json
import sympy
from sympy.parsing.sympy_parser import parse_expr

# --- CONFIGURAÇÕES ---
BASE_MODEL = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
ADAPTER_PATH = "qwen-rlvr-math-v1" # Nome da pasta do treino RLVR matemático
OUTPUT_FILE = "resultados_rlvr_gsm8k.jsonl"
DEVICE = "cuda"

print(f"Carregando Tokenizer e Base Model...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True
)
base_model.resize_token_embeddings(len(tokenizer))

print(f"Acoplando o modelo RLVR: {ADAPTER_PATH}...")
model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
model.eval()

# --- FUNÇÕES DE AVALIAÇÃO MATEMÁTICA ---
def extract_boxed_answer(text):
    match = re.search(r"\\boxed\{([^}]*)\}", text)
    return match.group(1).strip() if match else None

def is_math_equivalent(pred, truth):
    if pred is None: return False
    if pred == truth: return True
    pred_clean = pred.replace(",", "").strip()
    truth_clean = truth.replace(",", "").strip()
    try:
        if abs(float(pred_clean) - float(truth_clean)) < 1e-6: return True
    except: pass
    try:
        if sympy.simplify(parse_expr(pred_clean) - parse_expr(truth_clean)) == 0: return True
    except: pass
    return False

dataset = load_dataset("gsm8k", "main", split="test")

print(f"Iniciando avaliação RLVR em {len(dataset)} problemas...")
system_msg = "You are a mathematical reasoning assistant. Think step by step to solve the problem. You must put your final numerical answer inside \\boxed{}."

acertos = 0
resultados = []

for item in tqdm(dataset):
    question = item['question']
    truth_match = item['answer'].split('####')
    ground_truth = truth_match[1].strip() if len(truth_match) > 1 else ""
    
    prompt_text = (
        f"<|im_start|>system\n{system_msg}<|im_end|>\n"
        f"<|im_start|>user\n{question}\n<|im_end|>\n"
        f"<|im_start|>assistant\n" 
    )
    
    inputs = tokenizer(prompt_text, return_tensors="pt").to(DEVICE)
    
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=512, temperature=0.1, do_sample=True)
    
    raw_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    raw_completion = raw_output.split("assistant")[-1] if "assistant" in raw_output else raw_output
    
    predicted_answer = extract_boxed_answer(raw_completion)
    is_correct = is_math_equivalent(predicted_answer, ground_truth)
    
    if is_correct: acertos += 1
        
    resultados.append({
        "question": question,
        "ground_truth": ground_truth,
        "predicted_answer": predicted_answer,
        "is_correct": is_correct,
        "full_completion": raw_completion
    })

with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
    for r in resultados:
        f.write(json.dumps(r) + '\n')

acuracia = (acertos / len(dataset)) * 100
print("\n" + "="*40)
print(f"ACURÁCIA RLVR GSM8K: {acuracia:.2f}% ({acertos}/{len(dataset)})")
print("="*40)