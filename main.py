from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
import os
import json

app = FastAPI()

BASE_DIR = os.path.dirname(__file__)
model = joblib.load(os.path.join(BASE_DIR, "model.pkl"))
vectorizer = joblib.load(os.path.join(BASE_DIR, "vectorizer.pkl"))

class TextInput(BaseModel):
    text: str

@app.post("/predict")
def predict(data: TextInput):
    # support input multiple: "text1||text2||text3"
    texts = data.text.split("||")
    results = []

    for text in texts:
        vec = vectorizer.transform([text])
        pred = model.predict(vec)[0]
        prob = model.predict_proba(vec).max()

        results.append({
            "text": text,
            "sentiment": pred,
            "confidence": float(prob)
        })

    return {"results": results}
