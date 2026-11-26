
import argparse
import traceback

import torch
from pathlib import Path
from datasets import Dataset, concatenate_datasets, load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig
)
from peft import (
    LoraConfig,
    prepare_model_for_kbit_training
)
from trl import SFTConfig, SFTTrainer

from conversation_format import load_conversation_examples

parser = argparse.ArgumentParser(description="Fine-tune Lisa with LoRA.")
parser.add_argument(
    "--data",
    choices=["pairs", "conversations"],
    default="pairs",
    help="pairs: .datasets/*.jsonl made by convert_dataset.py (default). "
         "conversations: .conversations/<dataset>/<nnn>.jsonl made by "
         "convert_dataset.py --format conversations.",
)
args = parser.parse_args()

print("CUDA:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0))
print("VRAM:", torch.cuda.get_device_properties(0).total_memory / 1024**3)

MODEL_NAME = "Qwen/Qwen3-1.7B-Base"

# Cada exemplo é um turno: prompt (system + histórico + user) e completion
# (mood + answer). Se prompt + completion passar disso, o fim é cortado, e o fim
# é justamente o mood/answer que o modelo precisa aprender.
MAX_LENGTH = 2048

# Quantização de 4 bits
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True
)

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    dtype=torch.float16
)

model.config.use_cache = False

model = prepare_model_for_kbit_training(
    model,
    use_gradient_checkpointing=True
)

# Configuração LoRA
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj"
    ]
)

root = Path(__file__).resolve().parent

if args.data == "conversations":
    conversation_dir = root / ".conversations"
    files, examples = load_conversation_examples(conversation_dir)

    if not files:
        raise RuntimeError(
            f"Nenhum arquivo em {conversation_dir}. "
            "Rode: python convert_dataset.py --format conversations"
        )

    print(f"Conversas: {len(files)} arquivos em {conversation_dir.name}/")
    dataset = Dataset.from_list(examples)
else:
    dataset_dir = root / ".datasets"
    dataset_files = sorted(dataset_dir.glob("*.jsonl"))

    if not dataset_files:
        raise RuntimeError("Nenhum arquivo JSONL encontrado.")

    datasets = []

    for file in dataset_files:
        print(f"Carregando: {file.name}")

        current = load_dataset(
            "json",
            data_files=str(file),
            split="train"
        )

        datasets.append(current)

    dataset = concatenate_datasets(datasets)

dataset = dataset.filter(
    lambda ex: isinstance(ex.get("prompt"), str) and ex["prompt"].strip() != ""
    and isinstance(ex.get("completion"), str) and ex["completion"].strip() != ""
)
dataset = dataset.shuffle(seed=42)

# O TRL calcula a perda só na completion (mood + answer) e anexa o EOS a ela,
# então o modelo também aprende a parar depois do answer.
lengths = []
merged = 0

for ex in dataset:
    prompt_ids = tokenizer(ex["prompt"])["input_ids"]
    full_ids = tokenizer(ex["prompt"] + ex["completion"] + tokenizer.eos_token)["input_ids"]
    lengths.append(len(full_ids))
    if full_ids[:len(prompt_ids)] != prompt_ids:
        merged += 1

print(f"Exemplos: {len(lengths)} | tokens por exemplo: máx {max(lengths)}, limite {MAX_LENGTH}")

too_long = sum(1 for n in lengths if n > MAX_LENGTH)
if too_long:
    raise RuntimeError(
        f"{too_long} exemplo(s) passam de {MAX_LENGTH} tokens e seriam cortados "
        "(perdendo o mood/answer). Aumente MAX_LENGTH ou encurte os prompts."
    )

if merged:
    print(
        f"AVISO: em {merged} exemplo(s) os tokens se fundem na fronteira prompt/completion. "
        "O modelo treinaria num contexto diferente do que vê na inferência. "
        "O prompt deveria terminar em '\\n' (veja convert_dataset.py e inference.py)."
    )

training_config = SFTConfig(
    output_dir="./qwen-adapter",
    num_train_epochs=3,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=2e-4,
    warmup_steps=10,

    fp16=True,
    bf16=False,

    gradient_checkpointing=True,
    max_length=MAX_LENGTH,

    # Dataset prompt/completion: a perda vale só para o mood/answer.
    completion_only_loss=True,
    packing=False,

    logging_steps=10,
    save_steps=100,
    save_total_limit=2,

    optim="paged_adamw_8bit",
    report_to="none",
)

print(dataset.column_names)

try:
    trainer = SFTTrainer(
        model=model,
        args=training_config,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=lora_config,
    )
    
except Exception:
    traceback.print_exc()
    raise

for name, param in trainer.model.named_parameters():
    if param.requires_grad and param.dtype != torch.float32:
        param.data = param.data.to(torch.float32)

print("\n=== PRECISION DEBUG ===")
print("FP16:", trainer.args.fp16)
print("BF16:", trainer.args.bf16)

for name, param in model.named_parameters():
    if param.requires_grad:
        print(f"{name}: {param.dtype}, device={param.device}")

print("=======================\n")

trainer.train()

trainer.save_model("./qwen-adapter")
tokenizer.save_pretrained("./qwen-adapter")

print("Treinamento concluído!")
