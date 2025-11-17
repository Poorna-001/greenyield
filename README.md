# 🌱 GreenYield – Smart Crop Recommendation System

GreenYield is a modular, explainable AI platform that helps farmers and agronomists choose the most suitable crop based on soil and climate parameters. It combines a **React frontend**, **Flask backend**, and a **Random Forest ML model** trained on the Kaggle Crop Recommendation dataset.

---

## ⚙️ Tech stack

- **Frontend:** React.js
- **Backend:** Flask (Python)
- **ML:** scikit-learn (Random Forest)
- **Tooling:** npm, virtualenv, Docker (optional)
- **Explainability:** Feature importance visualization

---

## 🚀 Setup instructions

### 1) Clone the repository
```bash
git clone https://github.com/Poorna-001/greenyield.git
cd greenyield
```
### 2) Frontend(React)
```bash
cd frontend
npm install        # installs node_modules locally (not in repo)
npm start          # runs the app on http://localhost:3000
```
### 3) Backend (Flask)
```bash
cd backend
python -m venv venv

# Activate the virtual environment:
# Linux/Mac:
source venv/bin/activate
# Windows:
venv\Scripts\activate

pip install -r requirements.txt
python app/main.py    # runs the Flask API on http://localhost:5000
```
### 4) Connect frontend and backend
* The React app calls the Flask API endpoint POST /predict.

* Ensure the frontend is configured to point to http://localhost:5000 (check your API base URL in the frontend config).

* Submit soil and climate inputs; the backend responds with the recommended crop and feature importance.
###🔌 API overview
POST /predict
  * Request JSON:
    ```bash
    {
      "N": 90,
      "P": 42,
      "K": 43,
      "temperature": 22.0,
      "humidity": 80.0,
      "ph": 6.5,
      "rainfall": 120.0
    }
    ```
* Response JSON:
    ```bash
    {
      "recommended_crop": "rice",
      "feature_importance": {
      "N": 0.12,
      "P": 0.10,
      "K": 0.08,
      "temperature": 0.20,
      "humidity": 0.18,
      "ph": 0.15,
      "rainfall": 0.17
    }
  }
  
### 📊 Dataset
  * Source: Kaggle Crop Recommendation Dataset

  * Fields: N, P, K, temperature, humidity, pH, rainfall, crop

  * Used to train the Random Forest classifier and align with frontend input fields.

🌟 Features
  * Smart recommendations: Predicts the most suitable crop from soil and climate inputs.

  * Explainability: Displays feature importance to justify each prediction.

  * Clean UX: Simple forms, clear results, and graceful fallbacks for missing crop info.

  * Modular architecture: Frontend and backend decoupled; easy to extend and deploy.
🧪 Local testing
  * Frontend: npm test (if configured).

  * Backend: Use curl or Postman to send a POST request to http://localhost:5000/predict:
```bash
    curl -X POST http://localhost:5000/predict \
        -H "Content-Type: application/json" \
        -d '{"N":90,"P":42,"K":43,"temperature":22,"humidity":80,"ph":6.5,"rainfall":120}'
```
### 🔐 Notes on environment and Git
  * Do not commit dependencies: node_modules/ and Python venv/ are excluded by .gitignore.

  * Environment files: Keep secrets out of the repo. Use .env files locally and add them to .gitignore.

  * Reproducibility: Dependencies are captured in package.json/package-lock.json and requirements.txt.
