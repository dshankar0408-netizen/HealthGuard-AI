"""
content.py
Precautions, general advice, and disclaimers shown alongside each risk score.
Shared by app.py (Streamlit) and flask_app.py (Flask).
"""

DISEASE_CONTENT = {
    "heart": {
        "title": "Heart disease",
        "precautions": [
            "Don't smoke, and get support to quit if you do — it's the single biggest lever for heart risk.",
            "Keep blood pressure and cholesterol under regular review, not just when something feels wrong.",
            "Know your family history — early heart disease in close relatives changes your baseline risk.",
            "Treat new chest pain, pressure, or shortness of breath as urgent, not something to wait out.",
        ],
        "advice": [
            "Aim for ~150 minutes/week of moderate activity — brisk walking counts.",
            "Favor fiber-rich, low-saturated-fat meals; cut back on processed and fried food.",
            "Manage stress deliberately — sleep, downtime, and social connection all measurably help.",
            "Take prescribed medication consistently; don't stop or adjust it without your doctor.",
            "Get a lipid panel and blood pressure check at least annually, more often if at risk.",
        ],
    },
    "diabetes": {
        "title": "Diabetes",
        "precautions": [
            "Watch for excess thirst, frequent urination, fatigue, or blurred vision — get them checked.",
            "Maintain a healthy weight; even modest loss meaningfully lowers risk if you're above range.",
            "If diabetes runs in your family, ask about earlier or more frequent A1c screening.",
            "Untreated high blood sugar compounds — don't defer a first diagnosis conversation.",
        ],
        "advice": [
            "Favor low-glycemic, high-fiber meals over refined carbs and sugary drinks.",
            "Build in regular movement — even short walks after meals help blood sugar control.",
            "Get a fasting glucose or A1c test every 1-3 years, sooner with risk factors present.",
            "If prediabetic, structured lifestyle programs can meaningfully delay or prevent onset.",
        ],
    },
    "stroke": {
        "title": "Stroke",
        "precautions": [
            "High blood pressure is the leading modifiable risk factor for stroke — track it.",
            "Existing heart disease or atrial fibrillation raises stroke risk — keep it managed.",
            "Don't smoke — it roughly doubles stroke risk independent of other factors.",
            "Learn FAST: Face drooping, Arm weakness, Speech difficulty — Time to call emergency services.",
        ],
        "advice": [
            "Keep blood pressure and blood sugar in range through diet, activity, and medication.",
            "Limit sodium and alcohol; both push blood pressure up over time.",
            "Ask about screening for sleep apnea and atrial fibrillation if you snore heavily or feel palpitations.",
            "Stay physically active most days — even light daily activity lowers long-term risk.",
        ],
        "emergency": "Face drooping, Arm weakness, Speech difficulty — Time to call emergency services immediately. Stroke is time-critical; do not wait to see if symptoms pass.",
    },
    "kidney": {
        "title": "Kidney disease",
        "precautions": [
            "Diabetes and high blood pressure cause most chronic kidney disease — manage both tightly.",
            "Avoid regular, unsupervised use of NSAIDs (like ibuprofen) — they stress the kidneys over time.",
            "Ask for a creatinine/eGFR blood test if you have diabetes, hypertension, or a family history.",
            "Swelling in the legs, persistent fatigue, or foamy urine are worth mentioning to a doctor.",
        ],
        "advice": [
            "Reduce sodium and processed food to ease the load on your kidneys.",
            "Stay appropriately hydrated — ask a clinician what's right for your situation.",
            "Get routine kidney function labs if you're managing diabetes or hypertension.",
            "Keep blood pressure and blood glucose within the ranges your doctor sets for you.",
        ],
    },
}

DISCLAIMER = (
    "Each model is a logistic regression trained on a public clinical dataset for demonstration — "
    "not a validated diagnostic device. Risk scores can be wrong. Always consult a licensed "
    "clinician for real diagnosis, screening, or treatment decisions."
)


def risk_label(pct: float) -> str:
    if pct < 20:
        return "Low"
    if pct < 45:
        return "Moderate"
    return "High"


def risk_color(pct: float) -> str:
    if pct < 20:
        return "#3F7D5C"
    if pct < 45:
        return "#B8862E"
    return "#AE4B3F"
