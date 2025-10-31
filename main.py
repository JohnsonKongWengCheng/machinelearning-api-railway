# =====================================
# STEP 1: Import required libraries
# =====================================
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
DATA_URL = "https://github.com/JohnsonKongWengCheng/machinelearning-api-railway/blob/main/cleaned_keywords_dataset.csv.gz"

print("📦 Loading cleaned dataset...")
df = pd.read_csv(DATA_URL, compression="gzip")

# Keep only essential columns
df = df[['keywords', 'injury', 'age', 'sex']].dropna(subset=['keywords', 'injury'])

# Convert list-like string of keywords back into text
df['keywords'] = df['keywords'].apply(
    lambda x: ' '.join(eval(x)) if isinstance(x, str) and x.startswith('[') else str(x)
)

# Filter out invalid ages (1–100)
df = df[(df['age'] >= 1) & (df['age'] <= 100)]

print(f"✅ Dataset loaded with {len(df)} valid rows")

# =====================================
# STEP 3: Train ML model (TF-IDF + Naive Bayes)
# =====================================
X = df['keywords']
y = df['injury']

model = make_pipeline(TfidfVectorizer(), MultinomialNB())
model.fit(X, y)

# Save model for future use
joblib.dump(model, "injury_predictor.pkl")
print("✅ Model trained and saved successfully!")

# =====================================
# STEP 4: Initialize FastAPI app
# =====================================
app = FastAPI(title="Injury Prediction API (Railway)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow access from your frontend or Android app
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model (to ensure available even after Railway restart)
model = joblib.load("injury_predictor.pkl")

# =====================================
# STEP 5: Define input schema
# =====================================
class KeywordInput(BaseModel):
    keywords: list[str]

# =====================================
# STEP 6: Define prediction endpoint
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
    
    input_text = ' '.join(data.keywords)
    prediction = model.predict([input_text])[0]
    return {"keywords": data.keywords, "predicted_injury": prediction}

# =====================================
# STEP 7: Local testing entry point
# =====================================
if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting Injury Prediction API...")
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
