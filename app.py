"""Flask web application and JSON API for customer churn predictions."""

from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request
from joblib import load

from src.preprocess import model_feature_columns
from src.retention_assistant import RetentionAssistantError, generate_retention_plan

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "churn_pipeline.joblib"
load_dotenv(BASE_DIR / ".env")

CATEGORY_OPTIONS = {
    "gender": ["Female", "Male"],
    "Partner": ["Yes", "No"],
    "Dependents": ["Yes", "No"],
    "PhoneService": ["Yes", "No"],
    "MultipleLines": ["Yes", "No", "No phone service"],
    "InternetService": ["DSL", "Fiber optic", "No"],
    "OnlineSecurity": ["Yes", "No", "No internet service"],
    "OnlineBackup": ["Yes", "No", "No internet service"],
    "DeviceProtection": ["Yes", "No", "No internet service"],
    "TechSupport": ["Yes", "No", "No internet service"],
    "StreamingTV": ["Yes", "No", "No internet service"],
    "StreamingMovies": ["Yes", "No", "No internet service"],
    "Contract": ["Month-to-month", "One year", "Two year"],
    "PaperlessBilling": ["Yes", "No"],
    "PaymentMethod": [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ],
}
NUMERIC_FIELDS = {
    "SeniorCitizen": (0.0, 1.0),
    "tenure": (0.0, 72.0),
    "MonthlyCharges": (0.0, 200.0),
    "TotalCharges": (0.0, 20000.0),
}
REQUIRED_FIELDS = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "tenure", "PhoneService",
    "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
    "Contract", "PaperlessBilling", "PaymentMethod", "MonthlyCharges",
]

app = Flask(__name__)
app.config["SECRET_KEY"] = "local-development-key"


@lru_cache(maxsize=1)
def get_model() -> Any:
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            "The trained model is missing. Place the IBM Telco CSV in data/ and "
            "run `python -m src.train` before making predictions."
        )
    return load(MODEL_PATH)


def validate_customer(payload: Any) -> tuple[dict[str, Any] | None, dict[str, str]]:
    if not isinstance(payload, dict):
        return None, {"request": "Expected a JSON object or submitted form fields."}

    errors: dict[str, str] = {}
    unknown = set(payload) - set(REQUIRED_FIELDS) - {"TotalCharges"}
    for field in sorted(unknown):
        errors[field] = "Unexpected field."

    customer: dict[str, Any] = {}
    for field in REQUIRED_FIELDS:
        value = payload.get(field)
        if value is None or str(value).strip() == "":
            errors[field] = "This field is required."
            continue

        if field in CATEGORY_OPTIONS:
            normalized = str(value).strip()
            if normalized not in CATEGORY_OPTIONS[field]:
                errors[field] = f"Choose one of: {', '.join(CATEGORY_OPTIONS[field])}."
            else:
                customer[field] = normalized
            continue

        try:
            number = float(value)
        except (TypeError, ValueError):
            errors[field] = "Enter a valid number."
            continue
        minimum, maximum = NUMERIC_FIELDS[field]
        if not pd.notna(number) or number < minimum or number > maximum:
            errors[field] = f"Enter a number between {minimum:g} and {maximum:g}."
        elif field == "SeniorCitizen" and not number.is_integer():
            errors[field] = "Enter 0 or 1."
        else:
            customer[field] = int(number) if field == "SeniorCitizen" else number

    total_charges = payload.get("TotalCharges")
    if total_charges is None or str(total_charges).strip() == "":
        customer["TotalCharges"] = customer.get("MonthlyCharges", 0) * customer.get("tenure", 0)
    else:
        try:
            number = float(total_charges)
            minimum, maximum = NUMERIC_FIELDS["TotalCharges"]
            if not pd.notna(number) or number < minimum or number > maximum:
                errors["TotalCharges"] = f"Enter a number between {minimum:g} and {maximum:g}."
            else:
                customer["TotalCharges"] = number
        except (TypeError, ValueError):
            errors["TotalCharges"] = "Enter a valid number."

    return (customer if not errors else None), errors


def prediction_result(customer: dict[str, Any]) -> dict[str, Any]:
    model = get_model()
    row = pd.DataFrame([customer], columns=model_feature_columns(model))
    probability = float(model.predict_proba(row)[0, 1])
    churn = bool(model.predict(row)[0])

    if probability < 0.30:
        risk = "Low"
        recommendation = "Continue positive engagement and monitor the account during regular check-ins."
    elif probability < 0.60:
        risk = "Medium"
        recommendation = "Proactively check satisfaction, review service usage, and offer a relevant plan review."
    else:
        risk = "High"
        recommendation = "Prioritize a personal retention outreach; review pricing, contract options, and support needs."

    return {
        "churn_prediction": "Yes" if churn else "No",
        "churn_probability": round(probability, 4),
        "risk_level": risk,
        "retention_recommendation": recommendation,
    }


@app.get("/")
def index():
    return render_template("index.html", options=CATEGORY_OPTIONS, numeric_fields=NUMERIC_FIELDS)


@app.post("/predict")
def predict_form():
    customer, errors = validate_customer(request.form.to_dict())
    if errors:
        return render_template(
            "index.html", options=CATEGORY_OPTIONS, numeric_fields=NUMERIC_FIELDS,
            errors=errors, values=request.form,
        ), 400
    try:
        result = prediction_result(customer)
    except FileNotFoundError as error:
        return render_template(
            "index.html", options=CATEGORY_OPTIONS, numeric_fields=NUMERIC_FIELDS,
            model_error=str(error), values=request.form,
        ), 503
    return render_template("result.html", result=result, customer=customer)


@app.post("/api/predict")
def predict_api():
    payload = request.get_json(silent=True)
    if payload is None:
        return jsonify({"error": "Request body must be valid JSON with Content-Type: application/json."}), 400
    customer, errors = validate_customer(payload)
    if errors:
        return jsonify({"error": "Invalid customer data.", "details": errors}), 400
    try:
        return jsonify(prediction_result(customer))
    except FileNotFoundError as error:
        return jsonify({"error": str(error)}), 503


@app.post("/api/retention-plan")
def retention_plan_api():
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify({"error": "Request body must be a JSON object."}), 400

    customer, errors = validate_customer(payload.get("customer"))
    if errors:
        return jsonify({"error": "Invalid customer data.", "details": errors}), 400

    supplied_prediction = payload.get("prediction")
    if not isinstance(supplied_prediction, dict):
        return jsonify({"error": "A prediction object is required."}), 400

    try:
        # Recompute the prediction on the server; never trust client-supplied scores.
        prediction = prediction_result(customer)
    except FileNotFoundError as error:
        return jsonify({"error": str(error)}), 503

    try:
        plan = generate_retention_plan(customer, prediction)
    except RetentionAssistantError as error:
        return jsonify({"error": str(error)}), error.status_code
    return jsonify({"retention_plan": plan})


@app.get("/health")
def health():
    return jsonify({"status": "ok", "model_loaded": MODEL_PATH.is_file()})


if __name__ == "__main__":
    app.run(debug=False)
