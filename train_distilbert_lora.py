import os, json, numpy as np, pandas as pd, torch
from sklearn.model_selection import train_test_split
from datasets import Dataset, DatasetDict
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          DataCollatorWithPadding, Trainer, TrainingArguments)
from peft import LoraConfig, get_peft_model, TaskType
from sklearn.metrics import accuracy_score, f1_score, classification_report

CSV_PATH = "/content/caption_classify/Data.csv"            # put your CSV here (Drive or upload)
TEXT_COL = "query"
AUX_COL  = "image descriptions"
LABEL_COL= "Toxic Category"
CONCAT_BOTH = True

df = pd.read_csv(CSV_PATH)
df[TEXT_COL] = df[TEXT_COL].astype(str)
df[AUX_COL]  = df[AUX_COL].astype(str)
TEXT_USED = "text" if CONCAT_BOTH else TEXT_COL
if CONCAT_BOTH: df["text"] = df[TEXT_COL] + ", " + df[AUX_COL]

labels_sorted = sorted(df[LABEL_COL].astype(str).unique())
label2id = {name:i for i,name in enumerate(labels_sorted)}
id2label = {i:name for name,i in label2id.items()}
df["label_id"] = df[LABEL_COL].astype(str).map(label2id)

train_df, test_df = train_test_split(df, test_size=0.1, stratify=df["label_id"], random_state=42)
train_df, val_df  = train_test_split(train_df, test_size=0.1111, stratify=train_df["label_id"], random_state=42)

def to_ds(frame): return Dataset.from_pandas(frame[[TEXT_USED,"label_id"]].rename(columns={TEXT_USED:"text","label_id":"label"}), preserve_index=False)
ds = DatasetDict(train=to_ds(train_df), validation=to_ds(val_df), test=to_ds(test_df))

MODEL_NAME = "distilbert-base-uncased"; MAX_LEN = 128
tok = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
def preprocess(b):
    enc = tok(b["text"], truncation=True, max_length=MAX_LEN)
    enc["labels"] = b["label"]; return enc
ds_tok = ds.map(preprocess, batched=True, remove_columns=ds["train"].column_names)
collator = DataCollatorWithPadding(tok)

num_labels = len(label2id)
base = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME, num_labels=num_labels, id2label=id2label, label2id=label2id
)
lora_cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.1,
                      target_modules=["q_lin","v_lin"], bias="none", task_type=TaskType.SEQ_CLS)
model = get_peft_model(base, lora_cfg)

def compute_metrics(ev):
    logits, labels = ev
    preds = np.argmax(logits, axis=-1)
    return {"accuracy": accuracy_score(labels,preds), "f1": f1_score(labels,preds, average="weighted", zero_division=0)}

args = TrainingArguments(
    output_dir="./outputs_distilbert_lora", per_device_train_batch_size=16, per_device_eval_batch_size=16,
    learning_rate=2e-4, num_train_epochs=5, evaluation_strategy="epoch", save_strategy="epoch",
    load_best_model_at_end=True, metric_for_best_model="f1", greater_is_better=True,
    fp16=torch.cuda.is_available(), logging_steps=50
)

trainer = Trainer(model=model, args=args, train_dataset=ds_tok["train"], eval_dataset=ds_tok["validation"],
                  tokenizer=tok, data_collator=collator, compute_metrics=compute_metrics)

trainer.train()
metrics_val  = trainer.evaluate(ds_tok["validation"])
metrics_test = trainer.evaluate(ds_tok["test"])
print("Validation:", metrics_val); print("Test:", metrics_test)

preds = trainer.predict(ds_tok["test"])
y_true = preds.label_ids; import numpy as np
y_pred = np.argmax(preds.predictions, axis=-1)
print(classification_report(y_true, y_pred, target_names=[id2label[i] for i in range(len(id2label))]))

out_dir = "./outputs_distilbert_lora"
model.save_pretrained(os.path.join(out_dir, "lora_adapter"))
with open(os.path.join(out_dir, "label_maps.json"), "w") as f: json.dump({"id2label": id2label, "label2id": label2id}, f)
with open(os.path.join(out_dir, "metrics.json"), "w") as f: json.dump({"val": metrics_val, "test": metrics_test}, f, indent=2)
print(f"Saved artifacts to {out_dir}")
