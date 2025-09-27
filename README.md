# Caption + Classify App

This app does 3 things:
1. Captions an image using **BLIP** (image → text).
2. Classifies text (either your own input or the generated caption) with a **DistilBERT** model (fine-tuned with LoRA).
3. Saves everything to a **CSV file** (`Data.csv`) so you can view all past inputs and predictions.

---

## How to Run

### Local
```bash
# Make a virtual environment (Python 3.10+)
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# Install everything
pip install -r requirements.txt

# Start the app
streamlit run streamlit_app.py
