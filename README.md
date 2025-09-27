# Caption + Classify App

This app does 3 things:
1. Captions an image using BLIP (image → text).
2. Classifies text (either your own input or the generated caption) with a DistilBERT model (fine-tuned with LoRA).
3. Saves everything to a CSV file (Data.csv) so you can view all past inputs and predictions.

---

## How to Run

### Local
python -m venv .venv  
source .venv/bin/activate   # Windows: .venv\Scripts\activate  

pip install -r requirements.txt  

streamlit run streamlit_app.py  

Open the link Streamlit gives you (http://localhost:8501).

---

### Google Colab
!pip install -r /content/caption_classify/requirements.txt  

!pkill -f streamlit || true  
!python -m streamlit run /content/caption_classify/streamlit_app.py \  
    --server.port 8501 --server.address 0.0.0.0 >/content/st_log.txt 2>&1 &  

If you want a public link, install pyngrok and connect your ngrok token.

---

## Training (Optional)

python train_distilbert_lora.py \  
  --train_csv Data.csv \  
  --text_col text \  
  --label_col label \  
  --output_dir outputs_distilbert_lora  

The classifier will then load the LoRA weights from outputs_distilbert_lora/.

---

## Requirements

See requirements.txt:

streamlit  
pillow  
transformers  
accelerate  
bitsandbytes  
torch  
scikit-learn  
pandas  
numpy  
peft  

---

## Notes

imagecaption.py → handles BLIP captioning.  
classifier.py → handles text classification.  
Data.csv → stores all inputs + predictions.  
streamlit_app.py → the UI.  

---

## Suggested .gitignore

__pycache__/  
*.pt  
*.bin  
*.ckpt  
st_log.txt  
outputs_distilbert_lora/  
