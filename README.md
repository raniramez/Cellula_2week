# Caption + Classify App

This project combines **image captioning** and **text classification** in a Streamlit app.

- Upload an image → BLIP generates a caption.  
- Enter text or use the caption → DistilBERT (with optional LoRA) classifies it.  
- Results are saved automatically to **db.csv** (a log of all user inputs and predictions).  
- You can also keep a labeled dataset (**Data.csv**) for training or retraining the classifier.

---

## 📸 Screenshots / Demo

Add images or GIFs of the app here (recommended size: ~800px wide).

Example:

![Upload Example](assets/Screenshot 2025-09-27 030811.png)  
*Uploading an image to generate a caption.*

![Classification Example](assets/Screenshot 2025-09-27 031145.png)  
*Text classification result shown in the app.*

![Database Example](assets/Screenshot 2025-09-27 031151.png)  
*Viewing auto-saved predictions in db.csv.*

---

## 📂 Project Structure

