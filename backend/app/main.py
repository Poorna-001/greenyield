# backend/app/main.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import os

app = Flask(__name__)
CORS(app)

# -------- Config --------
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
MODEL_PATH = os.path.join(MODEL_DIR, "crop_model.pkl")
ENCODER_PATH = os.path.join(MODEL_DIR, "crop_encoder.pkl")
FEATURE_NAMES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]

# -------- Load model --------
try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    raise RuntimeError(f"Failed to load model at {MODEL_PATH}: {e}")

label_encoder = None
if os.path.exists(ENCODER_PATH):
    try:
        label_encoder = joblib.load(ENCODER_PATH)
    except Exception:
        label_encoder = None


def to_2d_array(payload: dict) -> np.ndarray:
    try:
        values = [float(payload[name]) for name in FEATURE_NAMES]
    except KeyError as missing:
        raise ValueError(f"Missing required field: {missing}")
    except (TypeError, ValueError):
        raise ValueError("All fields must be numeric.")
    return np.array([values], dtype=float)


def decode_label(pred):
    if label_encoder is not None:
        try:
            return label_encoder.inverse_transform([pred])[0]
        except Exception:
            return pred
    return pred


def get_feature_importance_contributions():
    importance = getattr(model, "feature_importances_", None)
    if importance is None or len(importance) != len(FEATURE_NAMES):
        return None
    return {FEATURE_NAMES[i]: round(float(importance[i]), 4) for i in range(len(FEATURE_NAMES))}


def compute_sustainability(features: dict, crop: str) -> int:
    """
    Compute a simple sustainability score (0-100).
    Rules are lightweight placeholders to illustrate the feature.
    """
    score = 100

    # pH suitability (neutral range favored)
    ph = features["ph"]
    if ph < 5.5 or ph > 8.0:
        score -= 20
    elif ph < 6.0 or ph > 7.5:
        score -= 10

    # Water requirement vs rainfall
    rainfall = features["rainfall"]
    if crop in ["rice", "sugarcane", "banana"]:
        if rainfall < 150:
            score -= 20
        elif rainfall < 120:
            score -= 30
    elif crop in ["pigeonpeas", "blackgram", "mungbean"]:
        if rainfall < 90:
            score += 10  # drought tolerant bonus

    # Temperature comfort zone (broad heuristic)
    temp = features["temperature"]
    if crop in ["wheat"]:
        if temp > 30 or temp < 10:
            score -= 15
    elif crop in ["mango", "papaya", "orange", "pomegranate", "maize", "rice"]:
        if temp < 18 or temp > 38:
            score -= 10

    # Nutrient balance (simple NPK guardrails)
    N, P, K = features["N"], features["P"], features["K"]
    if N < 10 or P < 10 or K < 10:
        score -= 10

    return max(0, min(int(round(score)), 100))


def top_k_alternatives(X: np.ndarray, k: int = 3):
    """Return top-k crops with confidence scores if model supports predict_proba."""
    if not hasattr(model, "predict_proba"):
        return []
    probs = model.predict_proba(X)[0]
    pairs = list(enumerate(probs))
    pairs.sort(key=lambda x: x[1], reverse=True)
    alts = []
    for idx, p in pairs[:k]:
        crop_name = decode_label(idx)
        alts.append({"crop": str(crop_name), "confidence": round(float(p), 3)})
    return alts


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200


@app.route("/predict", methods=["POST"])
def predict():
    """
    Expects JSON fields: N, P, K, temperature, humidity, ph, rainfall
    Returns:
    {
      "recommended_crop": string,
      "explanation": { feature: importance } | null,
      "features": { feature: value },
      "sustainability_score": number,
      "alternatives": [{ crop, confidence }]
    }
    """
    try:
        data = request.get_json(force=True)
        X = to_2d_array(data)

        pred = model.predict(X)[0]
        crop = decode_label(pred)

        contributions = get_feature_importance_contributions()
        features_dict = {name: float(X[0][i]) for i, name in enumerate(FEATURE_NAMES)}
        sustainability = compute_sustainability(features_dict, str(crop))
        alternatives = top_k_alternatives(X, k=3)

        return jsonify({
            "recommended_crop": str(crop),
            "explanation": contributions,
            "features": features_dict,
            # "sustainability_score": int(sustainability),
            "alternatives": alternatives
        }), 200

    except ValueError as ve:
        return jsonify({"error": str(ve)}), 400
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5050, debug=False)
