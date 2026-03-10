import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from human_eval.data import write_jsonl, read_problems
from tqdm import tqdm
import re

# --- CONFIGURAÇÕES ---
BASE_MODEL = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
ADAPTER_PATH = "qwen-rlvr-v17"  # Pasta onde seu treino foi salvo
OUTPUT_FILE = "samples_rlvr_v17.jsonl"
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
model.eval() # Modo de inferência (desliga dropout)

problems = read_problems()

def limpar_codigo_rlvr(prompt_original, completion):
    """
    Limpa a saída do modelo treinado com RLVR.
    O modelo V17 aprendeu a dar 4 espaços, mas às vezes repete a assinatura.
    """
    # 1. Remove Markdown
    if "```python" in completion:
        completion = completion.split("```python")[1].split("```")[0]
    elif "```" in completion:
        completion = completion.split("```")[1].split("```")[0]
    
    # 2. Remove espaços em branco extras no começo/fim geral
    completion = completion.strip("\n") 
    
    # 3. Verifica se o modelo repetiu a assinatura (ex: "def soma(a,b):")
    # O HumanEval já fornece o prompt, então precisamos remover a assinatura se ela aparecer
    match_def = re.search(r"def\s+(\w+)\s*\(", prompt_original)
    if match_def:
        nome_funcao = match_def.group(1)
        lines = completion.split('\n')
        
        # Se a primeira linha for a definição da função, removemos ela
        if len(lines) > 0 and f"def {nome_funcao}" in lines[0]:
            completion = "\n".join(lines[1:])
    
    return completion

samples = []
print(f"🚀 Iniciando geração para {len(problems)} problemas...")

# System Message usada no treino V17
system_msg = "You are a Python coding engine. Respond ONLY with the indented function body."

for task_id in tqdm(problems):
    problem = problems[task_id]
    
    # --- PROMPT V17 (Simulando o ambiente de treino) ---
    # O HumanEval dá o prompt até a assinatura.
    # Vamos formatar para parecer o treino.
    
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
    
    # Pega apenas a resposta do assistente
    if "assistant" in raw_output:
        raw_completion = raw_output.split("assistant")[-1]
    else:
        raw_completion = raw_output

    # Limpa e formata
    final_completion = limpar_codigo_rlvr(problem['prompt'], raw_completion)
    
    samples.append(dict(task_id=task_id, completion=final_completion))

# Salva o arquivo
write_jsonl(OUTPUT_FILE, samples)
print(f"✅ Arquivo gerado: {OUTPUT_FILE}")

# Tenta importar o avaliador
try:
    print("\n--- AVALIAÇÃO AUTOMÁTICA ---")
    from human_eval.evaluation import evaluate_functional_correctness
    
    results = evaluate_functional_correctness(
        sample_file=OUTPUT_FILE, 
        k=[1],
        n_workers=4,
        timeout=3.0
    )
    print(f"🏆 Resultado RLVR V17 (Pass@1): {results['pass@1'] * 100:.2f}%")
    
except ImportError:
    print("⚠️ Human-Eval não instalado ou não encontrado.")
    print(f"Rode manualmente: evaluate_functional_correctness {OUTPUT_FILE}")