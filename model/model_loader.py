"""Load and run the trained real-vs-AI image classifier once per process."""
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "artifacts" / "real_fake_classifier.joblib"


class ModelUnavailableError(RuntimeError):
    """Raised when the trained classifier cannot be loaded or used."""


@lru_cache(maxsize=1)
def get_model():
    """Return the persisted classifier, loading it only on first use."""
    if not MODEL_PATH.is_file():
        raise ModelUnavailableError(f"Trained model artifact is missing: {MODEL_PATH.name}")
    try:
        return joblib.load(MODEL_PATH)
    except Exception as error:
        raise ModelUnavailableError("The trained model artifact could not be loaded.") from error


def features(image_path):
    """Match the preprocessing used by the existing trained artifact exactly."""
    try:
        with Image.open(image_path) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
            small = image.resize((32, 32), Image.Resampling.LANCZOS)
            array = np.asarray(small, dtype=np.float32) / 255.0
    except (OSError, ValueError) as error:
        raise ModelUnavailableError("The saved image could not be prepared for prediction.") from error
    gray = array.mean(axis=2, keepdims=True)
    return np.concatenate([array, gray], axis=2).reshape(1, -1)


def predict_ai_probability(image_path):
    """Return p_ai from the artifact, where label 1 is the AI-generated class."""
    model = get_model()
    try:
        probabilities = model.predict_proba(features(image_path))[0]
        classes = [int(item) for item in model.classes_]
        return float(probabilities[classes.index(1)])
    except (AttributeError, IndexError, ValueError) as error:
        raise ModelUnavailableError("The trained model returned an invalid prediction.") from error
