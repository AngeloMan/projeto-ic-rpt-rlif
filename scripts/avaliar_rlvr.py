import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from human_eval.data import write_jsonl, read_problems
from tqdm import tqdm
import re
import sys

# --- CONFIGURAÇÕES ---
BASE_MODEL = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
ADAPTER_PATH = "qwen-rlvr" # Sua pasta do treino
OUTPUT_FILE = "samples_rlvr.jsonl"
DEVICE = "cuda"

print(f"🔄 Carregando Tokenizer: {BASE_MODEL}...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)

print(f"🔄 Carregando Modelo Base: {BASE_MODEL}...")
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)

# ⚠️ CRUCIAL: Redimensionar igual ao treino para o LoRA encaixar
print(f"📏 Ajustando embeddings para tamanho: {len(tokenizer)}...")
base_model.resize_token_embeddings(len(tokenizer))

print(f"🧠 Acoplando o Cérebro Treinado (LoRA): {ADAPTER_PATH}...")
model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
model.eval() # Modo de inferência

problems = read_problems()

# --- FUNÇÃO DE LIMPEZA (Herdada do seu script v3, que é muito boa) ---
def tratar_codigo(prompt_original, completion):
    # 1. Limpeza de Markdown
    if "```python" in completion:
        completion = completion.split("```python")[1].split("```")[0]
    elif "```" in completion:
        completion = completion.split("```")[1].split("```")[0]
    
    completion = completion.strip()
    
    # 2. Descobrir o nome da função (entry point)
    match_def = re.search(r"def\s+(\w+)\s*\(", prompt_original)
    nome_funcao = match_def.group(1) if match_def else None

    # 3. Lógica de Extração e Indentação
    linhas = completion.split('\n')
    
    # Caso A: O modelo reescreveu a assinatura da função ("def func():")
    if nome_funcao and any(nome_funcao in linha and "def " in linha for linha in linhas):
        corpo = []
        capturando = False
        indentacao_base = None
        
        for linha in linhas:
            if f"def {nome_funcao}" in linha:
                capturando = True
                continue
            
            if capturando:
                if indentacao_base is None and linha.strip():
                    indentacao_base = len(linha) - len(linha.lstrip())
                
                if indentacao_base is not None:
                    if len(linha) - len(linha.lstrip()) >= indentacao_base or linha.strip() == "":
                        corpo.append(linha[indentacao_base:])
                    else:
                        break
        
        completion_limpa = "\n".join(["    " + l for l in corpo])
        
    # Caso B: O modelo só cuspiu o código ("return x + y")
    else:
        # Se o modelo V17 já indentou (o que ele aprendeu a fazer), respeitamos.
        # Mas para garantir no HumanEval, forçamos 4 espaços se parecer plano.
        if linhas and not linhas[0].startswith("    "):
             completion_limpa = "\n".join(["    " + linha for linha in linhas])
        else:
             completion_limpa = completion

    return completion_limpa

samples = []
print(f"🚀 Iniciando geração para {len(problems)} problemas...")

# --- PROMPT V17 (O "Segredo" do seu treino) ---
# Usamos o System Prompt que ensinou ele a formatar
system_msg = "You are a Python coding engine. Respond ONLY with the indented function body."

for task_id in tqdm(problems):
    problem = problems[task_id]
    
    # Montamos o prompt simulando o ambiente de treino para ativar o LoRA
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
            temperature=0.1, # Baixa temperatura para precisão
            top_p=0.95,
            do_sample=True
        )
    
    raw_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Pega só a resposta do assistente
    if "assistant" in raw_output:
        raw_completion = raw_output.split("assistant")[-1]
    else:
        raw_completion = raw_output

    # Limpa e formata
    final_completion = tratar_codigo(problem['prompt'], raw_completion)
    
    samples.append(dict(task_id=task_id, completion=final_completion))

# Salva o arquivo
write_jsonl(OUTPUT_FILE, samples)
print(f"✅ Arquivo gerado: {OUTPUT_FILE}")

print(f"Rode manualmente: evaluate_functional_correctness {OUTPUT_FILE}")