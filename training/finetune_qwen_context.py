import json
import math
import os
import random
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq,
)
from peft import LoraConfig, get_peft_model, TaskType

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
DATA_FILE = "context_pairs.json"
SINGLE_ADAPTER = "qwen-lora-adapter"       
SYSTEM = ("Tu es un adulte bienveillant qui discute en français avec un enfant. "
          "Réponds simplement et naturellement.")
MAX_LEN = 512
RESERVE = 64   

tok = AutoTokenizer.from_pretrained(MODEL)
tok.padding_side = "right"

model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16)
model = get_peft_model(model, LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
))
model.print_trainable_parameters()

# ---- data: split by SOURCE FILE to prevent leaking ----
with open(DATA_FILE, encoding="utf-8") as f:
    data = json.load(f)

files = sorted({d["file"] for d in data})
random.Random(42).shuffle(files)
test_files = set(files[:max(1, round(len(files) * 0.1))])
train_rows = [d for d in data if d["file"] not in test_files]
test_rows = [d for d in data if d["file"] in test_files]
print(f"files: {len(files)}  train examples: {len(train_rows)}  test examples: {len(test_rows)}")


def build(ex):
    ctx = list(ex["context"])
    while True:
        msgs = [{"role": "system", "content": SYSTEM}] + ctx
        prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        prompt_ids = tok(prompt, add_special_tokens=False)["input_ids"]
        if len(prompt_ids) <= MAX_LEN - RESERVE or len(ctx) <= 1:
            break
        ctx = ctx[1:]   
    full = prompt + ex["reply"] + "<|im_end|>"
    full_ids = tok(full, add_special_tokens=False, truncation=True,
                   max_length=MAX_LEN)["input_ids"]
    labels = ([-100] * len(prompt_ids) + full_ids[len(prompt_ids):])[:len(full_ids)]
    return {"input_ids": full_ids,
            "attention_mask": [1] * len(full_ids),
            "labels": labels}


train_ds = Dataset.from_list(train_rows)
test_ds = Dataset.from_list(test_rows)
train_ds = train_ds.map(build, remove_columns=train_ds.column_names)
test_ds = test_ds.map(build, remove_columns=test_ds.column_names)

# ---- train ----
args = TrainingArguments(
    output_dir="qwen-lora-context",
    per_device_train_batch_size=4,     
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    num_train_epochs=2,
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
    eval_strategy="epoch",             
    save_strategy="epoch",
    logging_steps=25,
    fp16=True,
    report_to="none",
)
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=test_ds,
    data_collator=DataCollatorForSeq2Seq(tok, padding=True, label_pad_token_id=-100),
)
trainer.train()

metrics = trainer.evaluate()
print("Held-out loss:", metrics["eval_loss"])
print("Held-out perplexity:", math.exp(metrics["eval_loss"]))

model.save_pretrained("qwen-lora-context-adapter")
tok.save_pretrained("qwen-lora-context-adapter")

# ---- compare: base vs context-tuned ----
tok.padding_side = "left"
model.eval()

has_single = os.path.isdir(SINGLE_ADAPTER)
if has_single:
    model.load_adapter(SINGLE_ADAPTER, adapter_name="single")
model.set_adapter("default")

tests = [
    [("user", "Bonjour, comment ça va ?")],
    [("user", "J'ai faim.")],
    [("user", "Pourquoi le ciel est bleu ?")],
    [("user", "Tu veux jouer avec moi ?")],
    [("assistant", "qu'est-ce que tu as fait ce week-end ?"),
     ("user", "je suis allé à la piscine avec mon frère")],
    [("assistant", "tu aimes l'école ?"),
     ("user", "oui j'aime bien mais j'aime pas les maths")],
]


def reply(history):
    msgs = [{"role": "system", "content": SYSTEM}] + \
           [{"role": r, "content": c} for r, c in history]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = tok(text, return_tensors="pt", add_special_tokens=False).to(model.device)
    torch.manual_seed(0)   # same seed for every variant for fair comparison
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=60, do_sample=True,
                             temperature=0.7, top_p=0.9, repetition_penalty=1.1,
                             pad_token_id=tok.pad_token_id)
    return tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)


for history in tests:
    for r, c in history:
        print(("ENFANT : " if r == "user" else "ADULTE : ") + c)
    with model.disable_adapter():
        print("BASE   :", reply(history))
    if has_single:
        model.set_adapter("single")
        print("SINGLE :", reply(history))
    model.set_adapter("default")
    print("CONTEXT:", reply(history))
    print()