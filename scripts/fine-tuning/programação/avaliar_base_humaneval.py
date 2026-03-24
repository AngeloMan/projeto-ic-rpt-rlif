import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from human_eval.data import write_jsonl, read_problems
from tqdm import tqdm
import re

# --- CONFIGURAÇÕES ---
MODEL_NAME = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
OUTPUT_FILE = "samples_base.jsonl"
DEVICE = "cuda"

# --- SYSTEM PROMPT UNIFICADO ---
# Idêntico nos três scripts para garantir comparação justa.
SYSTEM_MSG = (
    "You are a Python coding engine. "
    "Respond with the COMPLETE Python function, including the def line and the indented body. "
    "Do NOT use input() or any form of user input. "
    "All values must come from function parameters. "
    "Do NOT include any explanation or markdown, only the raw Python function."
)

# --- FUNÇÃO DE PARSING UNIFICADA ---
# Idêntica nos três scripts.
def tratar_codigo(prompt_original, completion):
    # 1. Remove Markdown
    if "```python" in completion:
        completion = completion.split("```python")[1].split("```")[0]
    elif "```" in completion:
        completion = completion.split("```")[1].split("```")[0]

    completion = completion.strip()

    # 2. Descobre o nome da função pelo prompt do HumanEval
    match_def = re.search(r"def\s+(\w+)\s*\(", prompt_original)
    nome_funcao = match_def.group(1) if match_def else None

    linhas = completion.split('\n')

    # Caso A: modelo gerou a assinatura completa ("def func():")
    # Extrai só o corpo e reindenta com 4 espaços.
    if nome_funcao and any(nome_funcao in l and "def " in l for l in linhas):
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

    # Caso B: modelo gerou só o corpo sem assinatura
    else:
        if linhas and not linhas[0].startswith("    "):
            completion_limpa = "\n".join(["    " + l for l in linhas])
        else:
            completion_limpa = completion

    return completion_limpa

# --- CARREGAMENTO ---
print(f"Carregando {MODEL_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True,
)
model.eval()

problems = read_problems()
samples = []

print("Gerando amostras (greedy, temp=0.0)...")
for task_id in tqdm(problems):
    problem = problems[task_id]

    messages = [
        {"role": "system", "content": SYSTEM_MSG},
        {"role": "user",   "content": f"Implement:\n{problem['prompt']}"},
    ]
    text_input = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text_input, return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.0,   # greedy — padrão da literatura para pass@1
            do_sample=False,
        )

    raw_output = tokenizer.decode(outputs[0], skip_special_tokens=True)

    if "assistant" in raw_output:
        raw_completion = raw_output.split("assistant")[-1]
    else:
        raw_completion = raw_output

    final_completion = tratar_codigo(problem['prompt'], raw_completion)
    samples.append(dict(task_id=task_id, completion=final_completion))

write_jsonl(OUTPUT_FILE, samples)
print(f"Arquivo {OUTPUT_FILE} gerado com {len(samples)} amostras.")