# =====================================
# STEP 1: Import required libraries
# =====================================
import os
import ast
import pandas as pd
import joblib
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sklearn.pipeline import make_pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer

# =====================================
# STEP 2: Load your cleaned dataset (smaller + compressed)
# =====================================
DATA_URL = "https://github.com/JohnsonKongWengCheng/machinelearning-api-railway/raw/refs/heads/main/cleaned_keywords_dataset.csv.gz"

print("📦 Loading cleaned dataset...")
df = pd.read_csv(DATA_URL, compression="gzip")

# --- Ensure required columns exist ---
expected_cols = {"keywords", "injury"}
if not expected_cols.issubset(df.columns):
    raise ValueError(f"Dataset missing columns: {expected_cols - set(df.columns)}")

# Keep only essential columns safely
available_cols = [col for col in ["keywords", "injury", "age", "gender"] if col in df.columns]
df = df[available_cols].dropna(subset=["keywords", "injury"])

# Convert list-like strings (e.g., "['fall','ladder']") into clean text
def safe_eval_keywords(x):
    try:
        if isinstance(x, str) and x.startswith("["):
            return " ".join(ast.literal_eval(x))
        return str(x)
    except Exception:
        return str(x)

df["keywords"] = df["keywords"].apply(safe_eval_keywords)

# Filter out invalid ages (if column exists)
if "age" in df.columns:
    df = df[(df["age"].fillna(0) >= 1) & (df["age"].fillna(0) <= 100)]

print(f"✅ Dataset loaded with {len(df)} valid rows")

# =====================================
# STEP 3: Train ML model (TF-IDF + Naive Bayes)
# =====================================
X = df["keywords"]
y = df["injury"]

model = make_pipeline(TfidfVectorizer(), MultinomialNB())
model.fit(X, y)

# Save model
joblib.dump(model, "injury_predictor.pkl")
print("✅ Model trained and saved successfully!")

# =====================================
# STEP 4: Initialize FastAPI app
# =====================================
app = FastAPI(title="Injury Prediction API (Railway)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model (ensure available even after Railway restart)
if os.path.exists("injury_predictor.pkl"):
    model = joblib.load("injury_predictor.pkl")
else:
    print("⚠️ Model not found — retraining...")
    model = make_pipeline(TfidfVectorizer(), MultinomialNB())
    model.fit(X, y)
    joblib.dump(model, "injury_predictor.pkl")

# =====================================
# STEP 5: Define input schema
# =====================================
class KeywordInput(BaseModel):
    keywords: list[str]

# =====================================
# STEP 6: Root endpoint for testing
# =====================================
@app.get("/")
def root():
    return {"message": "✅ Injury Prediction API is running on Railway!"}

# =====================================
# STEP 7: Prediction endpoint
# =====================================
@app.post("/predict")
def predict_injury(data: KeywordInput):
    """
    Accepts list of keywords (from NLP API) and returns predicted injury.
    Example input:
    {
        "keywords": ["fall", "ladder", "unconscious"]
    }
    """
    if not data.keywords:
        return {"error": "No keywords provided."}

    input_text = " ".join(data.keywords)
    prediction = model.predict([input_text])[0]
    return {"keywords": data.keywords, "predicted_injury": prediction}

# =====================================
# STEP 8: Local testing entry point
# =====================================
if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting Injury Prediction API...")
    uvicorn.run("main:app", host="0.0.0.0", port=8000)
