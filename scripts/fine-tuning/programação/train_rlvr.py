import os
import sys
import torch
import re
import subprocess
import textwrap
import multiprocessing

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
OUTPUT_DIR = "qwen-rlvr-v2"

# --- 2. EXECUÇÃO SEGURA VIA SUBPROCESS ---
# Motivo da troca de multiprocessing para subprocess:
# No Windows (spawn), multiprocessing.Process re-importa o módulo __main__ inteiro
# para encontrar o _worker, executando patches de CUDA e imports pesados no processo
# filho — que crasha silenciosamente antes de colocar qualquer coisa na fila.
# subprocess.run spawna um Python limpo que só executa o código, sem nenhum import
# do módulo principal.

def run_code_safe(code_string, timeout=5.0):
    """
    Executa code_string em um processo Python completamente limpo via subprocess.
    Retorna True se o código rodar sem exceções, False se falhar ou timeout.
    """
    try:
        result = subprocess.run(
            [sys.executable, "-c", code_string],
            timeout=timeout,
            capture_output=True,
            text=True,
            stdin=subprocess.DEVNULL,   # [FIX BUG 1] input() recebe EOF imediato
        )                               # em vez de bloquear até o timeout
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        return False
    except Exception:
        return False

# --- 3. FUNÇÃO DE RECOMPENSA EXTERNA (RLVR) ---
def reward_rlvr_execution(completions, test_list, **kwargs):
    """
    A essência do RLVR: a recompensa vem do ambiente (testes unitários), não da probabilidade.
    """
    rewards = []

    signatures = kwargs.get("signature", [None] * len(completions))

    for idx, (completion, tests, sig) in enumerate(zip(completions, test_list, signatures)):
        # 1. Limpa a saída do modelo (remove Markdown)
        code = completion.strip()
        if "```python" in code:
            code = code.split("```python")[1].split("```")[0]
        elif "```" in code:
            code = code.split("```")[1].split("```")[0]

        code = textwrap.dedent(code).strip()

        # [FIX BUG 2] Fallback: se o modelo ainda gerar só o corpo sem def,
        # reconstrói a função completa usando a assinatura do dataset.
        if code and not re.match(r"^\s*(def |class |import |from )", code):
            if sig:
                indented_body = "\n".join("    " + line for line in code.splitlines())
                code = f"{sig}\n{indented_body}"

        # 2. Prepara o código com os testes do MBPP (ex: assert add(1,2) == 3)
        tests_str = "\n".join(tests)
        full_execution_code = f"{code}\n\n{tests_str}"

        # 3. Executa e recompensa
        passou = run_code_safe(full_execution_code)

        if passou:
            rewards.append(1.0)   # Recompensa positiva (feedback externo de sucesso)
        else:
            rewards.append(-1.0)  # Punição (erro de lógica, compilação ou loop infinito)

        # Monitor — mostra status do primeiro exemplo do batch
        if idx == 0:
            status = "✅ PASS" if passou else "❌ FAIL"
            print(f"\nMONITOR RLVR | {status}")
            print("-" * 50)
            print(code.strip()[:400])
            print("-" * 50 + "\n")

    return rewards

# --- 4. PREPARAÇÃO DOS DADOS ---
def format_data_func(example):
    gold_code = example.get('code', '')
    match = re.search(r"def\s+\w+\s*\(.*?\):", gold_code, re.DOTALL)
    signature = match.group(0) if match else "def solution():"

    prompt_raw = example.get('prompt', example.get('text', ''))

    # [FIX BUG 1] Proibir input() explicitamente no system prompt.
    # Sem isso, o modelo gerava funções com input() que bloqueavam o subprocess
    # esperando stdin que nunca chega, desperdiçando o timeout inteiro.
    # [FIX BUG 2] Pedir a função COMPLETA (def + corpo) em vez de só o corpo.
    # O prompt anterior ("ONLY with the indented function body") fazia o modelo
    # gerar apenas o corpo indentado sem o def — causando SyntaxError no exec.
    system_msg = (
        "You are a Python coding engine. "
        "Respond with the COMPLETE Python function, including the def line and the indented body. "
        "Do NOT use input() or any form of user input. "
        "All values must come from function parameters. "
        "Do NOT include any explanation or markdown, only the raw Python function."
    )

    prompt_text = (
        f"<|im_start|>system\n{system_msg}<|im_end|>\n"
        f"<|im_start|>user\nImplement:\n{signature}\n\"\"\"\n{prompt_raw}\n\"\"\"\n<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )

    return {
        "prompt": prompt_text,
        "test_list": example['test_list'],
        "signature": signature,   # guardamos a assinatura para o fallback abaixo
    }

# --- 5. EXECUÇÃO PRINCIPAL ---
def main():
    print(f"--- INICIANDO TREINO {OUTPUT_DIR} (RLVR v2 | subprocess | Verificador Externo) ---")

    # Seed global para reprodutibilidade entre runs
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    peft_config = LoraConfig(
        r=32,
        lora_alpha=64,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"],
        task_type=TaskType.CAUSAL_LM,
        lora_dropout=0.05,
    )

    training_args = GRPOConfig(
        output_dir=OUTPUT_DIR,
        learning_rate=5e-5,
        num_train_epochs=3,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        num_generations=4,
        max_prompt_length=600,
        max_completion_length=512,
        logging_steps=5,
        save_steps=50,
        fp16=True,
        beta=0.0005,
        use_vllm=False,
        report_to="none",
        temperature=0.7,
        seed=42,
        gradient_checkpointing=True,
        dataloader_pin_memory=False,
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

    print("Carregando Dataset (MBPP)...")
    dataset = load_dataset("mbpp", "sanitized", split="train")
    dataset = dataset.map(format_data_func, batched=False)

    print("Iniciando Trainer...")
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_rlvr_execution,
        args=training_args,
        train_dataset=dataset,
        peft_config=peft_config,
        processing_class=tokenizer,
    )

    print("TREINO RLVR INICIADO!")
    trainer.train()

    print("Salvando modelo final...")
    trainer.save_model(OUTPUT_DIR)
    print(f"Modelo RLVR salvo com sucesso em: {OUTPUT_DIR}")

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()