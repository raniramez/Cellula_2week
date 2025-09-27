# imagecaption.py — BLIP-2 FLAN-T5-XL only
import re
import torch
from PIL import Image
from transformers import Blip2Processor, Blip2ForConditionalGeneration

MODEL_NAME = "Salesforce/blip2-flan-t5-xl"
_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_PROC = None
_MODEL = None

def _lazy_load():
    global _PROC, _MODEL
    if _MODEL is not None:
        return
    _PROC = Blip2Processor.from_pretrained(MODEL_NAME)
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    _MODEL = Blip2ForConditionalGeneration.from_pretrained(
        MODEL_NAME,
        device_map="auto",
        torch_dtype=torch_dtype
    ).eval()

def preload():
    _lazy_load()

def _clean(text: str) -> str:
    """Clean up repetitions and spacing."""
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    # de-duplicate repeated phrases if separated by commas
    if "," in text and len(text) < 300:
        seen, dedup = set(), []
        for part in [p.strip() for p in text.split(",")]:
            if part.lower() not in seen:
                dedup.append(part)
                seen.add(part.lower())
        text = ", ".join(dedup)
    return text

def caption_image(
    img_path: str,
    prompt: str = "Write a detailed natural-language caption describing this photo.",
    max_new_tokens: int = 64,
    num_beams: int = 3,
    repetition_penalty: float = 1.08,
) -> str:
    """Generate a caption using BLIP-2 FLAN-T5-XL."""
    _lazy_load()
    image = Image.open(img_path).convert("RGB")

    # BLIP-2 prefers question/answer format
    q = f"Question: {prompt}\nAnswer:"
    inputs = _PROC(images=image, text=q, return_tensors="pt")
    device = next(_MODEL.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.inference_mode():
        out = _MODEL.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            repetition_penalty=repetition_penalty,
        )

    return _clean(_PROC.decode(out[0], skip_special_tokens=True).strip())
