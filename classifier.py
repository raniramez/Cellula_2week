import torch, json
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel

class ToxicClassifier:
    def __init__(self,
                 base_model="distilbert-base-uncased",
                 adapter_dir="./outputs_distilbert_lora/lora_adapter",
                 maps_path="./outputs_distilbert_lora/label_maps.json",
                 max_len=128):
        with open(maps_path,"r") as f:
            maps = json.load(f)
        self.id2label = {int(k): v for k, v in maps["id2label"].items()}
        self.label2id = maps["label2id"]
        self.max_len = max_len
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.tok = AutoTokenizer.from_pretrained(base_model, use_fast=True)
        base = AutoModelForSequenceClassification.from_pretrained(
            base_model,
            num_labels=len(self.label2id),
            id2label=self.id2label,
            label2id=self.label2id
        )
        self.model = PeftModel.from_pretrained(base, adapter_dir).to(self.device).eval()

    def classify(self, text: str):
        enc = self.tok(
            text, truncation=True, padding=True,
            max_length=self.max_len, return_tensors="pt"
        ).to(self.device)
        with torch.no_grad():
            logits = self.model(**enc).logits
            probs = F.softmax(logits, dim=-1)
        pred_id = int(torch.argmax(probs, dim=-1).item())
        score = float(probs[0, pred_id].item())
        return self.id2label[pred_id], score
