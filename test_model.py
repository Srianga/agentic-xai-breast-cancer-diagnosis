import joblib
import numpy as np

model = joblib.load("model/xgb_model.pkl")
scaler = joblib.load("model/scaler.pkl")
features = joblib.load("model/features.pkl")

print("Model loaded")
print("Scaler loaded")
print("Features loaded:", len(features))

x = np.random.rand(1, len(features))

x_scaled = scaler.transform(x)

prob = model.predict_proba(x_scaled)[0][1]
pred = int(prob >= 0.5)

print("Prediction:", pred)
print("Probability:", prob)

from dotenv import load_dotenv
import os

load_dotenv()

print("API KEY:", os.getenv("GOOGLE_API_KEY"))