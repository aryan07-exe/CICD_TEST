from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import os
import google.generativeai as genai
from dotenv import load_dotenv
load_dotenv()

# ------------------------------------
# Load ML components
# ------------------------------------
vectorizer = joblib.load("tfidf_vectorizer.pkl")
model = joblib.load("sentiment_model.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# ------------------------------------
# Initialize FastAPI
# ------------------------------------
app = FastAPI()

# ------------------------------------
# Load Gemini API Key from environment
# ------------------------------------
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)

# You can use Gemini Flash or Gemini Pro
llm = genai.GenerativeModel("gemini-2.5-flash")

# ------------------------------------
# Input Model
# ------------------------------------
class InputText(BaseModel):
    text: str


# ==========================================================
# 1️⃣ /predict-only → ML ONLY (for checking model performance)
# ==========================================================
@app.post("/predict-only")
def predict_only(request: InputText):

    vec = vectorizer.transform([request.text])
    pred_num = model.predict(vec)[0]
    pred_label = label_encoder.inverse_transform([pred_num])[0]

    return {
        "input_text": request.text,
        "predicted_sentiment": pred_label
    }


# ====================================================================
# 2️⃣ /predict-and-enhance → ML + Gemini LLM (for frontend usage)
# ====================================================================
@app.post("/predict-and-enhance")
def predict_and_enhance(request: InputText):

    # ML prediction
    vec = vectorizer.transform([request.text])
    pred_num = model.predict(vec)[0]
    pred_label = label_encoder.inverse_transform([pred_num])[0]

    # Gemini instruction
    prompt = f"""
    Analyze this text: "{request.text}"

    1. Explain why the sentiment is classified as **{pred_label}**.
    2. Provide a more polite and improved version of the text.

    And keep the response concise.nd in human language and only ive the text response no character
      or unnecessary charcter untill user asks or neededDo not include asterick or slash(*,/)
    """

    # Generate response from Gemini
    response = llm.generate_content(prompt)

    return {
        "input_text": request.text,
        "predicted_sentiment": pred_label,
        "enhanced_output": response.text
    }
