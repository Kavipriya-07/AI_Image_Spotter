"""Real prediction service backed by the persisted classifier artifact."""
from model.model_loader import ModelUnavailableError, predict_ai_probability

MODEL_NAME = "StandardScaler + LogisticRegression (real/AI classifier)"
THRESHOLD = 0.50


def predict_image(image_path):
    """Run the trained classifier and preserve the dashboard's response contract."""
    ai_probability = round(predict_ai_probability(image_path), 4)
    real_probability = round(1.0 - ai_probability, 4)
    is_ai = ai_probability >= THRESHOLD
    confidence = ai_probability if is_ai else real_probability
    return {
        "prediction": "AI-GENERATED" if is_ai else "REAL",
        "confidence": round(confidence, 4),
        "real_probability": real_probability,
        "ai_probability": ai_probability,
        "model_used": MODEL_NAME,
        "demo_indicators": [
            {"label": "AI-generated probability", "value": round(ai_probability * 100)},
            {"label": "Real probability", "value": round(real_probability * 100)},
            {"label": "Decision threshold", "value": round(THRESHOLD * 100)},
        ],
    }
