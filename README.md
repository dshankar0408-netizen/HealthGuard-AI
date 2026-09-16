# Disease Risk Predictor — Streamlit & Flask

Real scikit-learn models (StandardScaler + LogisticRegression), trained on four
public clinical datasets, wrapped in two interchangeable UIs. No JS
reimplementation of the model anywhere — both apps call the same pickled
`sklearn.pipeline.Pipeline` objects in `models/`.

| Disease | Dataset | Records | Test AUC | Accuracy |
|---|---|---|---|---|
| Heart disease | UCI Cleveland Heart Disease | 303 | 0.88 | 77% |
| Diabetes | Pima Indians Diabetes | 768 | 0.82 | 71% |
| Stroke | Stroke Prediction dataset | 5,110 | 0.84 | 95% |
| Kidney disease | UCI Chronic Kidney Disease | 400 | 0.97 | 91% |

## Setup

```bash
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Pretrained models are already included in `models/*.pkl`. To retrain from
scratch (e.g. after editing `train_and_export.py`):

```bash
python train_and_export.py
```

This reads the CSVs in `data/` and re-fits + re-pickles all four models.

## Run the Streamlit app (recommended — interactive sliders, live gauge)

```bash
streamlit run app.py
```

Opens at `http://localhost:8501`.

## Run the Flask app (REST API + plain HTML UI)

```bash
python flask_app.py
```

Opens at `http://127.0.0.1:5000`. Includes:
- A browsable UI at `/` and `/disease/<heart|diabetes|stroke|kidney>`
- A JSON API: `POST /api/predict/<disease>` with a JSON body of feature values,
  e.g.:

```bash
curl -X POST http://127.0.0.1:5000/api/predict/heart \
  -H "Content-Type: application/json" \
  -d '{"age":45,"sex":1,"trestbps":177,"chol":343,"fbs":0,"thalach":70,"exang":0,"oldpeak":0.5,"ca":0}'
```

`GET /api/diseases` lists every model's feature schema and metrics.

## Project structure

```
app.py                Streamlit UI
flask_app.py           Flask API + HTML UI
content.py             Shared precautions/advice/disclaimer text
train_and_export.py    Trains + pickles all 4 models from data/*.csv
templates/index.html   Flask UI template
models/*.pkl           Pretrained pipelines (StandardScaler + LogisticRegression)
models/registry.json   Feature schema + metrics per disease
data/*.csv             Source datasets
```

## Known data caveat

The heart-disease CSV mirror used here originally had an inverted label
column: `target=1` correlated with *healthier* profiles, not disease. This
is fixed in `train_and_export.py` (the label is flipped, documented inline)
— worth knowing if you swap in a different heart-disease CSV.

## Disclaimer

These are demonstration models trained on small-to-medium public datasets.
They are not validated diagnostic tools. Always consult a licensed
clinician for real diagnosis, screening, or treatment decisions.
