# =====================================
# STEP 1: Import required libraries
# =====================================
import re
import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.pipeline import make_pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer

# =====================================
# STEP 2: Load your dataset
# =====================================
data = pd.read_csv("FYP-ML-dataset.csv")
df = data.loc[:, ["Narrative_1", "Diagnosis"]].copy()
df.columns = ["scenario_raw", "injury"]

# =====================================
# STEP 3: Extract injury text (after DX:)
# =====================================
df.loc[:, 'injury_name'] = (
    df['scenario_raw']
    .str.extract(r'DX:\s*([A-Z\s]+[A-Z])', expand=False)
    .str.strip()
)

# =====================================
# STEP 4: Remove DX: part from scenario text
# =====================================
df.loc[:, 'scenario_clean'] = (
    df['scenario_raw']
    .str.replace(r'DX:.*', '', regex=True)
    .str.replace(r'\s+', ' ', regex=True)
    .str.strip()
)

# =====================================
# STEP 5: Extract structured info (age, gender)
# =====================================
def extract_age(text):
    match = re.search(r'(\d{1,2})\s*Y[O]?[MF]?', text, re.IGNORECASE)
    return int(match.group(1)) if match else None

def extract_gender(text):
    text = text.upper()
    if re.search(r'\bYOM\b|\bMALE\b', text):
        return 'Male'
    elif re.search(r'\bYOF\b|\bFEMALE\b', text):
        return 'Female'
    return None

df.loc[:, 'age'] = df['scenario_clean'].apply(extract_age)
df.loc[:, 'gender'] = df['scenario_clean'].apply(extract_gender)
df = df.assign(age=df['age'].astype('Int64'))

# =====================================
# STEP 6: Clean up scenario text (remove age/gender tags)
# =====================================
def clean_scenario(text):
    text = re.sub(r'\b\d{1,2}\s*Y[O]?[MF]?\b', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\b(MALE|FEMALE)\b', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

df.loc[:, 'scenario_clean'] = df['scenario_clean'].apply(clean_scenario)

# =====================================
# STEP 7: Rename and reorder columns
# =====================================
df = df.rename(columns={'scenario_clean': 'scenario'})
df = df[['scenario', 'injury_name', 'injury', 'age', 'gender']]

# =====================================
# STEP 8: Extract keywords for ML model
# =====================================
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

def extract_keywords(text):
    words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
    words = [w for w in words if w not in ENGLISH_STOP_WORDS]
    return ' '.join(words)

df['keywords'] = df['scenario'].apply(extract_keywords)

# =====================================
# STEP 9: Train the ML model (TF-IDF + Naive Bayes)
# =====================================
X = df['keywords']
y = df['injury_name'].fillna('Unknown')

model = make_pipeline(TfidfVectorizer(), MultinomialNB())
model.fit(X, y)

# =====================================
# STEP 10: Save trained model
# =====================================
joblib.dump(model, "injury_predictor.pkl")

# =====================================
# STEP 11: Create FastAPI app
# =====================================
app = FastAPI(title="Injury Prediction API")

# Load model
model = joblib.load("injury_predictor.pkl")

class KeywordInput(BaseModel):
    keywords: list[str]

@app.post("/predict")
def predict_injury(data: KeywordInput):
    """
    Accepts list of keywords (from NLP API) and returns predicted injury
    Example input:
    {
        "keywords": ["fall", "ladder", "no", "response"]
    }
    """
    input_text = ' '.join(data.keywords)
    prediction = model.predict([input_text])[0]
    return {"prediction": prediction}

# =====================================
# STEP 12: Run the app (local testing)
# =====================================
if __name__ == "__main__":
    import uvicorn
    print("✅ Dataset loaded and model trained successfully.")
    print(df.head(10))
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
