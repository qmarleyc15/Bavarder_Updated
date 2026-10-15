import json
import math
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
PAIRS_FILE = "/content/conversation_pairs_final.json"
SYSTEM = ("Tu es un adulte bienveillant qui discute en français avec un enfant. "
          "Réponds simplement et naturellement.")
MAX_LEN = 160

tok = AutoTokenizer.from_pretrained(MODEL)
tok.padding_side = "right"  # right-pad for training

model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16)
model = get_peft_model(model, LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
))
model.print_trainable_parameters()

# ---- data: 90/10 split, chat format, loss only on the adult reply ----
with open(PAIRS_FILE, encoding="utf-8") as f:
    pairs = json.load(f)
ds = Dataset.from_list(pairs).train_test_split(test_size=0.1, seed=42)


def build(ex):
    msgs = [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": ex["child"]}]
    prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    full = prompt + ex["adult"] + "<|im_end|>"
    prompt_ids = tok(prompt, add_special_tokens=False)["input_ids"]
    full_ids = tok(full, add_special_tokens=False, truncation=True,
                   max_length=MAX_LEN)["input_ids"]
    labels = ([-100] * len(prompt_ids) + full_ids[len(prompt_ids):])[:len(full_ids)]
    return {"input_ids": full_ids,
            "attention_mask": [1] * len(full_ids),
            "labels": labels}


ds = ds.map(build, remove_columns=["child", "adult"])
print(f"train: {len(ds['train'])}  test: {len(ds['test'])}")

# ---- train ----
args = TrainingArguments(
    output_dir="qwen-lora",
    per_device_train_batch_size=6,    
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=2,
    learning_rate=2e-4,
    num_train_epochs=3,
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
    train_dataset=ds["train"],
    eval_dataset=ds["test"],
    data_collator=DataCollatorForSeq2Seq(tok, padding=True, label_pad_token_id=-100),
)
trainer.train()

metrics = trainer.evaluate()
print("Held-out loss:", metrics["eval_loss"])
print("Held-out perplexity:", math.exp(metrics["eval_loss"]))

model.save_pretrained("qwen-lora-adapter")
tok.save_pretrained("qwen-lora-adapter")

# ---- compare base model vs fine-tuned results on the same prompts ----
tok.padding_side = "left"
model.eval()
prompts = [
    "Bonjour, comment ça va ?",
    "Qu'est-ce que tu fais aujourd'hui ?",
    "J'ai faim.",
    "Pourquoi le ciel est bleu ?",
    "Tu veux jouer avec moi ?",
]


def reply(p):
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": p}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = tok(text, return_tensors="pt", add_special_tokens=False).to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=60, do_sample=True,
                             temperature=0.7, top_p=0.9, repetition_penalty=1.1,
                             pad_token_id=tok.pad_token_id)
    return tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)


for p in prompts:
    with model.disable_adapter():
        base = reply(p)
    tuned = reply(p)
    print(f"IN   : {p}\nBASE : {base}\nTUNED: {tuned}\n")