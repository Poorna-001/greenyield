import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import joblib

# 1. Load dataset
df = pd.read_csv("data/raw/Crop_recommendation.csv")

# 2. Normalize crop labels
df['label'] = df['label'].str.strip().str.lower()

# 3. Encode crop labels
crop_encoder = LabelEncoder()
df['crop_encoded'] = crop_encoder.fit_transform(df['label'])

# 4. Define features (X) and target (y)
X = df[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
y = df['crop_encoded']

# 5. Train model
model = RandomForestClassifier(random_state=42)
model.fit(X, y)

# 6. Save model and encoder
joblib.dump(model, "backend/app/models/crop_model.pkl")
joblib.dump(crop_encoder, "backend/app/models/crop_encoder.pkl")

print("✅ Crop recommendation model and encoder saved successfully!")
