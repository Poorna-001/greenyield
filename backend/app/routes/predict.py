from flask import Blueprint, request, jsonify
import joblib
import os

predict_bp = Blueprint("predict", __name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "..", "models")

model = joblib.load(os.path.join(MODEL_DIR, "crop_model.pkl"))
crop_encoder = joblib.load(os.path.join(MODEL_DIR, "crop_encoder.pkl"))

@predict_bp.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(force=True)

        required_fields = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
        missing = [f for f in required_fields if f not in data]
        if missing:
            return jsonify({"error": f"Missing fields: {', '.join(missing)}"}), 400

        features = [
            int(data["N"]),
            int(data["P"]),
            int(data["K"]),
            float(data["temperature"]),
            float(data["humidity"]),
            float(data["ph"]),
            float(data["rainfall"])
        ]

        prediction = model.predict([features])[0]
        crop = crop_encoder.inverse_transform([prediction])[0]

        response = jsonify({"recommended_crop": crop})
        response.headers.add("Access-Control-Allow-Origin", "*")
        return response

    except Exception as e:
        return jsonify({"error": str(e)}), 400
