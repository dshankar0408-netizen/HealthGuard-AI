"""
flask_app.py — Disease Risk Predictor (Flask)

Same real trained scikit-learn models as app.py, exposed as:
  - a JSON REST API:  POST /api/predict/<disease>
  - a simple HTML UI:  GET  /  and  GET /disease/<disease>

Run:
    python flask_app.py
Then open http://127.0.0.1:5000
"""
import pickle
import json
from pathlib import Path

from flask import Flask, request, jsonify, render_template
import pandas as pd

from content import DISEASE_CONTENT, DISCLAIMER, risk_label, risk_color

BASE = Path(__file__).parent
MODELS_DIR = BASE / "models"

app = Flask(__name__)

REGISTRY = json.load(open(MODELS_DIR / "registry.json"))
MODELS = {}
for name in REGISTRY:
    with open(MODELS_DIR / f"{name}.pkl", "rb") as f:
        MODELS[name] = pickle.load(f)


def run_inference(disease, payload):
    bundle = MODELS[disease]
    pipeline = bundle["pipeline"]
    feature_order = bundle["feature_order"]
    meta = REGISTRY[disease]

    row = [float(payload.get(k, meta["features"][k]["min"])) for k in feature_order]
    X = pd.DataFrame([row], columns=feature_order)

    proba = float(pipeline.predict_proba(X)[0, 1])
    pct = min(98, max(1, proba * 100))

    scaler = pipeline.named_steps["scaler"]
    clf = pipeline.named_steps["clf"]
    z_contribs = clf.coef_[0] * scaler.transform(X)[0]
    contributions = [
        {"factor": meta["features"][k]["label"], "contribution": round(float(c), 4)}
        for k, c in zip(feature_order, z_contribs)
    ]
    contributions = sorted(
        [c for c in contributions if c["contribution"] > 0.05],
        key=lambda c: -c["contribution"],
    )[:6]

    return {
        "disease": disease,
        "risk_pct": round(pct, 1),
        "risk_label": risk_label(pct),
        "risk_color": risk_color(pct),
        "contributions": contributions,
        "model_info": {
            "n_samples": meta["n_samples"],
            "source": meta["source"],
            "auc": meta["auc"],
            "accuracy": meta["accuracy"],
        },
    }


@app.route("/api/diseases")
def api_diseases():
    return jsonify({k: {"title": DISEASE_CONTENT[k]["title"], **REGISTRY[k]} for k in REGISTRY})


@app.route("/api/predict/<disease>", methods=["POST"])
def api_predict(disease):
    if disease not in MODELS:
        return jsonify({"error": f"unknown disease '{disease}'"}), 404
    payload = request.get_json(force=True, silent=True) or {}
    try:
        result = run_inference(disease, payload)
    except (ValueError, KeyError) as e:
        return jsonify({"error": str(e)}), 400
    return jsonify(result)


@app.route("/")
def index():
    return render_template(
        "index.html",
        diseases=REGISTRY,
        titles={k: DISEASE_CONTENT[k]["title"] for k in REGISTRY},
        selected="heart",
        meta=REGISTRY["heart"],
        content=DISEASE_CONTENT["heart"],
        disclaimer=DISCLAIMER,
        result=None,
    )


@app.route("/disease/<disease>", methods=["GET", "POST"])
def disease_page(disease):
    if disease not in MODELS:
        return "Unknown disease", 404

    meta = REGISTRY[disease]
    content = DISEASE_CONTENT[disease]
    result = None

    if request.method == "POST":
        payload = {k: request.form.get(k, 0) for k in meta["features"]}
        result = run_inference(disease, payload)

    return render_template(
        "index.html",
        diseases=REGISTRY,
        titles={k: DISEASE_CONTENT[k]["title"] for k in REGISTRY},
        selected=disease,
        meta=meta,
        content=content,
        disclaimer=DISCLAIMER,
        result=result,
    )


if __name__ == "__main__":
    app.run(debug=True)
