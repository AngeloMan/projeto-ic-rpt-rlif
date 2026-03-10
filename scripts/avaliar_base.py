import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from human_eval.data import write_jsonl, read_problems
from tqdm import tqdm
import re

MODEL_NAME = "Qwen/Qwen2.5-Coder-1.5B-Instruct" 
OUTPUT_FILE = "amostras_base.jsonl"
DEVICE = "cuda"

print(f"Carregando {MODEL_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, 
    torch_dtype=torch.float16, 
    device_map="auto"
)

problems = read_problems()

def tratar_codigo(prompt_original, completion):
    # 1. Limpeza de Markdown
    if "```python" in completion:
        completion = completion.split("```python")[1].split("```")[0]
    elif "```" in completion:
        completion = completion.split("```")[1].split("```")[0]
    
    completion = completion.strip()
    
    # 2. Descobrir o nome da função (entry point)
    # O prompt do HumanEval geralmente termina com "def nome_funcao(...):"
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
            # Se achou a definição, começa a prestar atenção
            if f"def {nome_funcao}" in linha:
                capturando = True
                continue # Pula a linha do def
            
            if capturando:
                # Detecta a indentação da primeira linha útil
                if indentacao_base is None and linha.strip():
                    indentacao_base = len(linha) - len(linha.lstrip())
                
                # Só adiciona se fizer parte do corpo (tem indentação maior ou igual)
                if indentacao_base is not None:
                    if len(linha) - len(linha.lstrip()) >= indentacao_base or linha.strip() == "":
                        # Remove a indentação original para padronizar
                        corpo.append(linha[indentacao_base:])
                    else:
                        break # Acabou a função
        
        # Reconstrói o corpo e aplica indentação de 4 espaços
        completion_limpa = "\n".join(["    " + l for l in corpo])
        
    # Caso B: O modelo só cuspiu o código ("return x + y") sem indentação
    else:
        # Simplesmente adiciona 4 espaços em cada linha
        completion_limpa = "\n".join(["    " + linha for linha in linhas])

    return completion_limpa

samples = []
print("Gerando e corrigindo indentação...")

for task_id in tqdm(problems):
    problem = problems[task_id]
    
    # Prompt mais direto
    messages = [
        {"role": "system", "content": "You are a coding assistant. Write the body of the function directly."},
        {"role": "user", "content": f"Complete this Python code:\n{problem['prompt']}"}
    ]
    
    text_input = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text_input, return_tensors="pt").to(DEVICE)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=512, 
            temperature=0.0, 
            do_sample=False
        )
    
    raw_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Pega só a resposta do assistente
    if "assistant" in raw_output:
        raw_completion = raw_output.split("assistant")[-1]
    else:
        raw_completion = raw_output

    # AQUI ESTÁ A MÁGICA
    final_completion = tratar_codigo(problem['prompt'], raw_completion)
    
    samples.append(dict(task_id=task_id, completion=final_completion))

write_jsonl(OUTPUT_FILE, samples)
print(f"Arquivo {OUTPUT_FILE} gerado.")