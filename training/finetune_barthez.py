import json
import math
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)
from peft import LoraConfig, get_peft_model, TaskType

MODEL = "moussaKam/barthez"  # French BART, ~165M params

tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL)

lora = LoraConfig(
    task_type=TaskType.SEQ_2_SEQ_LM,
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "v_proj"],
)
model = get_peft_model(model, lora)
model.print_trainable_parameters()

# ---- data: 90/10 train/test split ----
with open("/content/conversation_pairs_filtered.json", encoding="utf-8") as f:
    pairs = json.load(f)

ds = Dataset.from_list(pairs).train_test_split(test_size=0.1, seed=42)


def prep(batch):
    enc = tok(batch["child"], max_length=64, truncation=True)
    enc["labels"] = tok(text_target=batch["adult"], max_length=96, truncation=True)["input_ids"]
    return enc


ds = ds.map(prep, batched=True, remove_columns=["child", "adult"])

# ---- train ----
args = Seq2SeqTrainingArguments(
    output_dir="bavarder-lora",
    per_device_train_batch_size=8,  
    per_device_eval_batch_size=16,
    learning_rate=3e-4,
    num_train_epochs=8,
    eval_strategy="epoch",            
    save_strategy="epoch",
    logging_steps=50,
    fp16=True,                        
    report_to="none",
)

trainer = Seq2SeqTrainer(
    model=model,
    args=args,
    train_dataset=ds["train"],
    eval_dataset=ds["test"],
    data_collator=DataCollatorForSeq2Seq(tok, model=model),
)

trainer.train()

# ---- evaluate on held-out pairs ----
metrics = trainer.evaluate()
print("Held-out loss:", metrics["eval_loss"])
print("Held-out perplexity:", math.exp(metrics["eval_loss"]))

model.save_pretrained("bavarder-lora-adapter")
tok.save_pretrained("bavarder-lora-adapter")

# ---- sample generations  ----
model.eval()
prompts = [
    "Bonjour, comment ça va ?",
    "Qu'est-ce que tu fais aujourd'hui ?",
    "J'ai faim.",
    "Pourquoi le ciel est bleu ?",
    "Tu veux jouer avec moi ?",
]

for p in prompts:
    inputs = tok(p, return_tensors="pt").to(model.device)
    out = model.generate(**inputs, max_new_tokens=40, num_beams=4,
                         no_repeat_ngram_size=3, encoder_no_repeat_ngram_size=2)
    print(f"IN : {p}\nOUT: {tok.decode(out[0], skip_special_tokens=True)}\n")