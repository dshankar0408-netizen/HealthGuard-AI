"""
train_and_export.py
Trains the four disease-risk models on real public clinical datasets and
pickles the full scikit-learn Pipeline (StandardScaler + LogisticRegression)
for use by app.py (Streamlit) and flask_app.py (Flask API).

Run once:
    python train_and_export.py

Requires data/heart.csv, data/diabetes.csv, data/stroke.csv, data/kidney.csv
(already included in this package). Re-download instructions are in README.md.
"""
import pandas as pd
import numpy as np
import pickle
import json
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score

DATA = Path(__file__).parent / "data"
MODELS = Path(__file__).parent / "models"
MODELS.mkdir(exist_ok=True)

registry = {}


def fit_export(name, X, y, feature_meta, description, source):
    X = X.astype(float)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000, C=1.0)),
    ])
    pipe.fit(Xtr, ytr)

    auc = roc_auc_score(yte, pipe.predict_proba(Xte)[:, 1])
    acc = accuracy_score(yte, pipe.predict(Xte))

    with open(MODELS / f"{name}.pkl", "wb") as f:
        pickle.dump({"pipeline": pipe, "feature_order": list(X.columns)}, f)

    registry[name] = {
        "description": description,
        "source": source,
        "n_samples": int(len(X)),
        "auc": round(float(auc), 3),
        "accuracy": round(float(acc), 3),
        "features": feature_meta,
    }
    print(f"{name:10s} n={len(X):5d}  AUC={auc:.3f}  ACC={acc:.3f}  -> models/{name}.pkl")


# ---------------- Heart disease ----------------
h = pd.read_csv(DATA / "heart.csv", encoding="utf-8-sig")
h.columns = [c.strip() for c in h.columns]
feat = ["age", "sex", "trestbps", "chol", "fbs", "thalach", "exang", "oldpeak", "ca"]
X = h[feat].copy()
# In this CSV mirror, target=1 correlates with *healthier* profiles (verified via
# feature correlations), i.e. the label is inverted vs. the conventional
# "1 = disease present" meaning. Flip it so 1 = elevated clinical risk.
y = 1 - h["target"].astype(int)
meta = {
    "age": {"label": "Age", "min": 25, "max": 80, "step": 1, "unit": "yrs", "kind": "slider"},
    "sex": {"label": "Male", "min": 0, "max": 1, "step": 1, "unit": "", "kind": "toggle"},
    "trestbps": {"label": "Resting blood pressure", "min": 90, "max": 200, "step": 1, "unit": "mmHg", "kind": "slider"},
    "chol": {"label": "Cholesterol", "min": 120, "max": 400, "step": 1, "unit": "mg/dL", "kind": "slider"},
    "fbs": {"label": "Fasting blood sugar >120", "min": 0, "max": 1, "step": 1, "unit": "", "kind": "toggle"},
    "thalach": {"label": "Max heart rate achieved", "min": 70, "max": 210, "step": 1, "unit": "bpm", "kind": "slider"},
    "exang": {"label": "Exercise-induced angina", "min": 0, "max": 1, "step": 1, "unit": "", "kind": "toggle"},
    "oldpeak": {"label": "ST depression (oldpeak)", "min": 0.0, "max": 6.0, "step": 0.1, "unit": "", "kind": "slider"},
    "ca": {"label": "Major vessels colored (0-3)", "min": 0, "max": 3, "step": 1, "unit": "", "kind": "slider"},
}
fit_export("heart", X, y, meta, "Estimates likelihood of significant coronary disease.", "UCI Cleveland Heart Disease dataset")

# ---------------- Diabetes ----------------
cols = ["pregnancies", "glucose", "bp", "skin", "insulin", "bmi", "pedigree", "age", "outcome"]
d = pd.read_csv(DATA / "diabetes.csv", header=None, names=cols)
feat = ["pregnancies", "glucose", "bp", "bmi", "pedigree", "age"]
X = d[feat].copy()
y = d["outcome"].astype(int)
meta = {
    "pregnancies": {"label": "Pregnancies", "min": 0, "max": 15, "step": 1, "unit": "", "kind": "slider"},
    "glucose": {"label": "Plasma glucose", "min": 70, "max": 200, "step": 1, "unit": "mg/dL", "kind": "slider"},
    "bp": {"label": "Diastolic blood pressure", "min": 50, "max": 120, "step": 1, "unit": "mmHg", "kind": "slider"},
    "bmi": {"label": "BMI", "min": 15.0, "max": 50.0, "step": 0.1, "unit": "", "kind": "slider"},
    "pedigree": {"label": "Diabetes pedigree (family history)", "min": 0.0, "max": 2.5, "step": 0.01, "unit": "", "kind": "slider"},
    "age": {"label": "Age", "min": 18, "max": 85, "step": 1, "unit": "yrs", "kind": "slider"},
}
fit_export("diabetes", X, y, meta, "Estimates likelihood of type 2 diabetes.", "Pima Indians Diabetes dataset (UCI)")

# ---------------- Stroke ----------------
s = pd.read_csv(DATA / "stroke.csv", encoding="utf-8-sig")
s.columns = [c.strip().lower() for c in s.columns]
s["bmi"] = pd.to_numeric(s["bmi"], errors="coerce")
s["bmi"] = s["bmi"].fillna(s["bmi"].mean())
s["ever_smoked"] = s["smoking_status"].isin(["formerly smoked", "smokes"]).astype(int)
feat = ["age", "hypertension", "heart_disease", "avg_glucose_level", "bmi", "ever_smoked"]
X = s[feat].copy()
y = s["stroke"].astype(int)
meta = {
    "age": {"label": "Age", "min": 1, "max": 90, "step": 1, "unit": "yrs", "kind": "slider"},
    "hypertension": {"label": "Hypertension", "min": 0, "max": 1, "step": 1, "unit": "", "kind": "toggle"},
    "heart_disease": {"label": "Existing heart disease", "min": 0, "max": 1, "step": 1, "unit": "", "kind": "toggle"},
    "avg_glucose_level": {"label": "Average glucose level", "min": 55, "max": 280, "step": 1, "unit": "mg/dL", "kind": "slider"},
    "bmi": {"label": "BMI", "min": 15.0, "max": 50.0, "step": 0.1, "unit": "", "kind": "slider"},
    "ever_smoked": {"label": "Smokes / formerly smoked", "min": 0, "max": 1, "step": 1, "unit": "", "kind": "toggle"},
}
fit_export("stroke", X, y, meta, "Estimates stroke likelihood from vascular and lifestyle factors.", "Stroke Prediction dataset (5,110 records)")

# ---------------- Kidney disease ----------------
k = pd.read_csv(DATA / "kidney.csv")
k.columns = [c.strip().lower() for c in k.columns]
for c in k.columns:
    if k[c].dtype == object:
        k[c] = k[c].astype(str).str.strip().replace({"?": np.nan, "nan": np.nan, "\t?": np.nan})
num_cols = ["age", "bp", "bgr", "sc", "hemo"]
for c in num_cols:
    k[c] = pd.to_numeric(k[c], errors="coerce")
    k[c] = k[c].fillna(k[c].mean())
k["htn"] = (k["htn"] == "yes").astype(int)
k["dm"] = (k["dm"] == "yes").astype(int)
k["ane"] = (k["ane"] == "yes").astype(int)
k["classification"] = k["classification"].str.replace("ckd\t", "ckd", regex=False).str.strip()
y = (k["classification"] == "ckd").astype(int)
feat = ["age", "bp", "bgr", "sc", "hemo", "htn", "dm", "ane"]
X = k[feat].copy()
meta = {
    "age": {"label": "Age", "min": 2, "max": 90, "step": 1, "unit": "yrs", "kind": "slider"},
    "bp": {"label": "Blood pressure", "min": 50, "max": 180, "step": 1, "unit": "mmHg", "kind": "slider"},
    "bgr": {"label": "Blood glucose (random)", "min": 70, "max": 490, "step": 1, "unit": "mg/dL", "kind": "slider"},
    "sc": {"label": "Serum creatinine", "min": 0.4, "max": 15.0, "step": 0.1, "unit": "mg/dL", "kind": "slider"},
    "hemo": {"label": "Hemoglobin", "min": 3.0, "max": 17.8, "step": 0.1, "unit": "g/dL", "kind": "slider"},
    "htn": {"label": "Hypertension", "min": 0, "max": 1, "step": 1, "unit": "", "kind": "toggle"},
    "dm": {"label": "Diabetes mellitus", "min": 0, "max": 1, "step": 1, "unit": "", "kind": "toggle"},
    "ane": {"label": "Anemia", "min": 0, "max": 1, "step": 1, "unit": "", "kind": "toggle"},
}
fit_export("kidney", X, y, meta, "Estimates chronic kidney disease likelihood from blood and vital measurements.", "UCI Chronic Kidney Disease dataset")

with open(MODELS / "registry.json", "w") as f:
    json.dump(registry, f, indent=2)
print("\nSaved models/registry.json")
