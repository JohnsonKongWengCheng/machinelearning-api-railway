# =====================================
# STEP 1: Import required libraries
# =====================================
import os
import joblib
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# =====================================
# STEP 2: Initialize FastAPI app
# =====================================
app = FastAPI(title="Injury Prediction API (Railway)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =====================================
# STEP 3: Load pretrained model
# =====================================
MODEL_PATH = "injury_predictor.pkl"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError("❌ injury_predictor.pkl not found. Please upload it to Railway.")

model = joblib.load(MODEL_PATH)
print("✅ Model loaded successfully!")

# =====================================
# STEP 4: Input schema
# =====================================
class KeywordInput(BaseModel):
    keywords: list[str]

# =====================================
# STEP 5: Root endpoint (for testing)
# =====================================
@app.get("/")
def root():
    return {"message": "✅ Injury Prediction API is running on Railway!"}

# =====================================
# STEP 6: Prediction endpoint
# =====================================
@app.post("/predict")
def predict_injury(data: KeywordInput):
    if not data.keywords:
        return {"error": "No keywords provided."}

    input_text = " ".join(data.keywords)
    prediction = model.predict([input_text])[0]
    return {"keywords": data.keywords, "predicted_injury": prediction}

# =====================================
# STEP 7: Local run
# =====================================
if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting API locally...")
    uvicorn.run("main:app", host="0.0.0.0", port=8000)
