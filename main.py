from fastapi import FastAPI, UploadFile, File
import pandas as pd
import joblib
import json
import numpy as np

app = FastAPI()

# Load model dan vectorizer
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")


# ============================
# 🔥 1. PREDIKSI TEKS BIASA
# ============================
@app.post("/predict")
def predict_multi_text(data: dict):
    input_text = data.get("text", "")

    # Pisahkan teks berdasarkan koma
    texts = [t.strip() for t in input_text.split(",") if t.strip()]

    if not texts:
        return {"error": "Input text kosong atau format salah"}

    results = []

    for text in texts:
        vec = vectorizer.transform([text])
        pred = model.predict(vec)[0]
        prob = float(model.predict_proba(vec).max())

        results.append({
            "text": text,
            "sentiment": pred,
            "confidence": prob
        })

    return results


# ============================
# 🔥 2. PREDIKSI FILE CSV
# ============================
@app.post("/predict-csv")
async def predict_csv(file: UploadFile = File(...)):
    if file.content_type != "text/csv":
        return {"error": "File must be a CSV"}

    # Baca CSV ke DataFrame
    df = pd.read_csv(file.file)

    if "review" not in df.columns:
        return {"error": "CSV harus memiliki kolom 'review'"}

    texts = df["review"].tolist()
    X = vectorizer.transform(texts)
    preds = model.predict(X)

    summary = {"positif": 0, "negatif": 0, "netral": 0}

    results = []
    for text, label in zip(texts, preds):
        results.append({
            "text": text,
            "sentiment": label
        })
        summary[label] += 1

    return {
        "results": results,
        "summary": summary
    }
