import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
import tensorflow as tf
import pickle
import numpy as np
import re
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ---------------------------
# Initialize FastAPI app
# ---------------------------
app = FastAPI()

# ---------------------------
# Load trained model
# ---------------------------
model = tf.keras.models.load_model("model.h5")

# ---------------------------
# Load tokenizer
# ---------------------------
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

# ---------------------------
# Model input settings
# question1 (5) + question2 (5) = 10
# ---------------------------
MAX_LEN = 5

# ---------------------------
# Input schema
# ---------------------------
class InputText(BaseModel):
    question1: str
    question2: str

# ---------------------------
# Text cleaning function
# ---------------------------
def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9 ]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# ---------------------------
# Preprocessing function
# ---------------------------
def preprocess(q1: str, q2: str):
    seq1 = tokenizer.texts_to_sequences([q1])
    seq2 = tokenizer.texts_to_sequences([q2])

    seq1 = pad_sequences(seq1, maxlen=MAX_LEN)
    seq2 = pad_sequences(seq2, maxlen=MAX_LEN)

    return np.concatenate([seq1, seq2], axis=1)

# ---------------------------
# Prediction endpoint
# ---------------------------
@app.post("/predict")
def predict_text(data: InputText):
    # Clean input
    q1 = clean_text(data.question1)
    q2 = clean_text(data.question2)

    # ✅ HARD RULE: identical questions
    if q1 == q2 and q1 != "":
        return {
            "prediction": "Duplicate",
            "score": 1.0
        }

    # Model prediction
    input_data = preprocess(q1, q2)
    prediction = model.predict(input_data)[0][0]

    # ✅ Slightly lower threshold
    label = "Duplicate" if prediction >= 0.4 else "Not Duplicate"

    return {
        "prediction": label,
        "score": float(prediction)
    }

# ---------------------------
# Run server
# ---------------------------
if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="127.0.0.1",
        port=8000,
        reload=True
    )
