import os, sys, json, glob, pickle, subprocess
from pathlib import Path

ROOT = Path.cwd()
DATA = ROOT / "dataset" / "data"
ART = ROOT / "artifacts"
ART.mkdir(exist_ok=True)

def install(pkg, import_name=None):
    import_name = import_name or pkg.split("==")[0].replace("-", "_")
    try:
        __import__(import_name)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])

# Lightweight stack - no TensorFlow download
for p, n in [
    ("pyarrow", "pyarrow"),
    ("numpy", "numpy"),
    ("pillow", "PIL"),
    ("scikit-learn", "sklearn"),
    ("joblib", "joblib"),
    ("streamlit", "streamlit"),
]:
    install(p, n)

import numpy as np
import pyarrow.parquet as pq
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import joblib

print("\n========================================")
print(" AI IMAGE REAL/FAKE DETECTION PROJECT")
print("========================================\n")

files = sorted(DATA.glob("*.parquet"))
if not files:
    raise FileNotFoundError("No parquet files found in dataset/data")

print("Parquet files:", len(files))
for f in files:
    print("  ", f.name, round(f.stat().st_size/1024/1024, 1), "MB")

X = []
y = []

def decode_image(obj):
    if obj is None:
        return None

    # pyarrow struct -> dict
    if isinstance(obj, dict):
        b = obj.get("bytes")
        p = obj.get("path")

        if b is not None:
            try:
                return Image.open(__import__("io").BytesIO(b)).convert("RGB")
            except Exception:
                pass

        if p:
            candidates = [
                ROOT / str(p),
                DATA / str(p),
                ROOT / "dataset" / str(p),
            ]
            for c in candidates:
                if c.exists():
                    try:
                        return Image.open(c).convert("RGB")
                    except Exception:
                        pass

    if isinstance(obj, (bytes, bytearray)):
        try:
            return Image.open(__import__("io").BytesIO(obj)).convert("RGB")
        except Exception:
            pass

    return None

print("\nReading dataset...")

for idx, f in enumerate(files, 1):
    table = pq.read_table(f, columns=["image", "label"])
    rows = table.to_pylist()

    good = 0
    for row in rows:
        img = decode_image(row["image"])
        if img is None:
            continue

        try:
            # Compact representation for fast CPU training
            img = img.resize((32, 32), Image.Resampling.LANCZOS)
            arr = np.asarray(img, dtype=np.float32) / 255.0

            # RGB + grayscale-like channel statistics
            gray = arr.mean(axis=2, keepdims=True)
            feat = np.concatenate([arr, gray], axis=2).reshape(-1)

            X.append(feat)
            y.append(int(row["label"]))
            good += 1
        except Exception:
            continue

    print(f"  {idx}/{len(files)} {f.name}: {good} usable images")

X = np.asarray(X, dtype=np.float32)
y = np.asarray(y, dtype=np.int64)

if len(X) < 100:
    raise RuntimeError(f"Only {len(X)} usable images found. Dataset image decoding failed.")

print("\nUsable images:", len(X))
print("Feature size:", X.shape[1])
print("Label distribution:", dict(zip(*np.unique(y, return_counts=True))))

# Preserve class balance
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining model...")
model = Pipeline([
    ("scale", StandardScaler()),
    ("classifier", LogisticRegression(
        max_iter=300,
        solver="lbfgs",
        random_state=42
    ))
])

model.fit(X_train, y_train)

pred = model.predict(X_test)
prob = model.predict_proba(X_test)

acc = accuracy_score(y_test, pred)
prec = precision_score(y_test, pred, average="weighted", zero_division=0)
rec = recall_score(y_test, pred, average="weighted", zero_division=0)
f1 = f1_score(y_test, pred, average="weighted", zero_division=0)
cm = confusion_matrix(y_test, pred)

metrics = {
    "test_samples": int(len(y_test)),
    "accuracy": float(acc),
    "precision": float(prec),
    "recall": float(rec),
    "f1_score": float(f1),
    "confusion_matrix": cm.tolist(),
    "classes": [int(x) for x in model.classes_],
}

joblib.dump(model, ART / "real_fake_classifier.joblib")
with open(ART / "metrics.json", "w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=2)

with open(ART / "model_info.json", "w", encoding="utf-8") as f:
    json.dump({
        "model": "StandardScaler + LogisticRegression",
        "image_size": "32x32 RGB + grayscale",
        "train_samples": int(len(y_train)),
        "test_samples": int(len(y_test)),
        "dataset_files": [f.name for f in files],
        "classes": [int(x) for x in model.classes_]
    }, f, indent=2)

print("\n========================================")
print(" ACTUAL TEST RESULTS")
print("========================================")
print(f"Accuracy : {acc:.4f} ({acc*100:.2f}%)")
print(f"Precision: {prec:.4f}")
print(f"Recall   : {rec:.4f}")
print(f"F1 Score : {f1:.4f}")
print("\nConfusion Matrix:")
print(cm)
print("\nClassification Report:")
print(classification_report(y_test, pred, zero_division=0))

# ---------------------------------------------------------
# Prediction helper
# ---------------------------------------------------------
(ROOT / "predict.py").write_text(r'''
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
''', encoding="utf-8")

# ---------------------------------------------------------
# Streamlit dashboard
# ---------------------------------------------------------
(ROOT / "app.py").write_text(r'''
import json
from pathlib import Path
import joblib
import numpy as np
import streamlit as st
from PIL import Image

st.set_page_config(
    page_title="AI Image Real/Fake Detector",
    page_icon="🔍",
    layout="wide"
)

ROOT = Path(__file__).parent
MODEL = joblib.load(ROOT / "artifacts" / "real_fake_classifier.joblib")

with open(ROOT / "artifacts" / "metrics.json", encoding="utf-8") as f:
    metrics = json.load(f)

st.title("🔍 AI Image Real/Fake Detector")
st.caption("Machine-learning based image classification dashboard")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Accuracy", f"{metrics['accuracy']*100:.2f}%")
c2.metric("Precision", f"{metrics['precision']*100:.2f}%")
c3.metric("Recall", f"{metrics['recall']*100:.2f}%")
c4.metric("F1 Score", f"{metrics['f1_score']*100:.2f}%")

st.divider()

uploaded = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png", "webp"]
)

if uploaded:
    img = Image.open(uploaded).convert("RGB")

    left, right = st.columns(2)

    with left:
        st.image(img, caption="Uploaded image", use_container_width=True)

    with right:
        small = img.resize((32,32), Image.Resampling.LANCZOS)
        arr = np.asarray(small, dtype=np.float32) / 255.0
        gray = arr.mean(axis=2, keepdims=True)
        x = np.concatenate([arr, gray], axis=2).reshape(1,-1)

        pred = int(MODEL.predict(x)[0])
        probs = MODEL.predict_proba(x)[0]
        classes = [int(c) for c in MODEL.classes_]
        confidence = float(max(probs))

        # Dataset label convention
        label = {0: "REAL", 1: "FAKE"}.get(pred, str(pred))

        st.subheader("Prediction")

        if label == "FAKE":
            st.error(f"🚨 {label}")
        else:
            st.success(f"✅ {label}")

        st.metric("Confidence", f"{confidence*100:.2f}%")

        st.write("Class probabilities")
        for cls, p in zip(classes, probs):
            name = {0: "REAL", 1: "FAKE"}.get(cls, str(cls))
            st.progress(float(p), text=f"{name}: {p*100:.2f}%")

st.divider()

with st.expander("Model details"):
    st.write("Model: StandardScaler + LogisticRegression")
    st.write("Input representation: 32×32 RGB + grayscale features")
    st.write(f"Training samples: {metrics['test_samples'] and 'see artifacts/model_info.json'}")
    st.json(metrics)
''', encoding="utf-8")

(ROOT / "requirements.txt").write_text(
"""numpy
pyarrow
pillow
scikit-learn
joblib
streamlit
""", encoding="utf-8")

(ROOT / "README.md").write_text(
"""# AI Image Real/Fake Detection

## Project
Machine-learning based classification of images into REAL and FAKE classes.

## Dataset
The project reads the downloaded Hugging Face parquet dataset from:

`dataset/data/`

## Model
StandardScaler + LogisticRegression using 32x32 RGB and grayscale image features.

## Training
The training script:
- reads all parquet files
- decodes image bytes/paths
- performs stratified train/test split
- trains the classifier
- calculates accuracy, precision, recall and F1
- saves the trained model

## Generated files

- `artifacts/real_fake_classifier.joblib`
- `artifacts/metrics.json`
- `artifacts/model_info.json`
- `predict.py`
- `app.py`
- `requirements.txt`

## Dashboard

Run:

`streamlit run app.py`

Then upload JPG/PNG/WEBP images and view predictions.

## Important
All reported metrics are calculated from the actual downloaded dataset and held-out test split. No fabricated metrics are used.
""", encoding="utf-8")

print("\n========================================")
print(" PROJECT READY")
print("========================================")
print("Model       :", ART / "real_fake_classifier.joblib")
print("Metrics     :", ART / "metrics.json")
print("Prediction  : predict.py")
print("Dashboard   : app.py")
print("README      : README.md")
print("\nStarting dashboard...")
print("Press Ctrl+C in this terminal to stop it.\n")

subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py"])
