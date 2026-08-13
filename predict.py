
import sys, json, joblib
import numpy as np
from PIL import Image

MODEL = joblib.load("artifacts/real_fake_classifier.joblib")

def features(path):
    img = Image.open(path).convert("RGB").resize((32,32), Image.Resampling.LANCZOS)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    gray = arr.mean(axis=2, keepdims=True)
    return np.concatenate([arr, gray], axis=2).reshape(1,-1)

if len(sys.argv) != 2:
    print("Usage: python predict.py <image_path>")
    raise SystemExit(1)

x = features(sys.argv[1])
pred = int(MODEL.predict(x)[0])
proba = MODEL.predict_proba(x)[0]

classes = [int(c) for c in MODEL.classes_]
confidence = float(max(proba))

# Dataset convention: labels are reported directly.
label_name = {0: "REAL", 1: "FAKE"}
result = label_name.get(pred, str(pred))

print(json.dumps({
    "prediction": result,
    "label": pred,
    "confidence": round(confidence, 4)
}, indent=2))
