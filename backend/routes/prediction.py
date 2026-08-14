"""JSON APIs for prediction, history, statistics, and model information."""
from datetime import datetime, timezone
import json, os, sqlite3, time, uuid
from pathlib import Path
from flask import Blueprint, current_app, jsonify, request, send_from_directory
from werkzeug.utils import secure_filename
from backend.services.image_service import ImageValidationError, inspect_image
from backend.services.prediction_service import MODEL_NAME, ModelUnavailableError, predict_image

prediction_bp = Blueprint("prediction", __name__, url_prefix="/api")
DATABASE = Path("history.db")


def db_connection():
    connection = sqlite3.connect(DATABASE)
    connection.row_factory = sqlite3.Row
    connection.execute("""CREATE TABLE IF NOT EXISTS analyses (id INTEGER PRIMARY KEY AUTOINCREMENT, filename TEXT NOT NULL, image_url TEXT NOT NULL, file_type TEXT NOT NULL, file_size INTEGER NOT NULL, resolution TEXT NOT NULL, prediction TEXT NOT NULL, confidence REAL NOT NULL, real_probability REAL NOT NULL, ai_probability REAL NOT NULL, analyzed_at TEXT NOT NULL, model_used TEXT NOT NULL)""")
    return connection


def history_rows():
    with db_connection() as connection:
        rows = connection.execute("SELECT * FROM analyses ORDER BY id DESC").fetchall()
    return [dict(row) | {"status": "Completed"} for row in rows]


@prediction_bp.post("/predict")
def predict():
    image = request.files.get("image")
    try:
        metadata = inspect_image(image)
        safe_name = secure_filename(image.filename)
        unique_name = f"{uuid.uuid4().hex}_{safe_name}"
        upload_dir = Path(current_app.config["UPLOAD_FOLDER"])
        upload_dir.mkdir(exist_ok=True)
        saved_path = upload_dir / unique_name
        image.save(saved_path)
        started = time.perf_counter()
        result = predict_image(saved_path)
        analysis_time = round(time.perf_counter() - started, 3)
        record = {"filename": safe_name, "image_url": f"/uploads/{unique_name}", "file_type": metadata["file_type"], "file_size": os.path.getsize(saved_path), "resolution": metadata["resolution"], "prediction": result["prediction"], "confidence": result["confidence"], "real_probability": result["real_probability"], "ai_probability": result["ai_probability"], "analyzed_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"), "model_used": result["model_used"]}
        with db_connection() as connection:
            cursor = connection.execute("""INSERT INTO analyses (filename,image_url,file_type,file_size,resolution,prediction,confidence,real_probability,ai_probability,analyzed_at,model_used) VALUES (:filename,:image_url,:file_type,:file_size,:resolution,:prediction,:confidence,:real_probability,:ai_probability,:analyzed_at,:model_used)""", record)
            record["id"] = cursor.lastrowid
        return jsonify({"status": "success", **result, **metadata, **record, "analysis_time": analysis_time})
    except ImageValidationError as error:
        return jsonify({"status": "error", "message": str(error)}), 400
    except ModelUnavailableError as error:
        current_app.logger.exception("Model unavailable: %s", error)
        return jsonify({"status": "error", "message": "The trained model is currently unavailable. Please try again later."}), 503
    except Exception:
        return jsonify({"status": "error", "message": "Analysis could not be completed. Please try another image."}), 500


@prediction_bp.get("/history")
def get_history():
    return jsonify({"status": "success", "history": history_rows()})


@prediction_bp.delete("/history")
def clear_history():
    with db_connection() as connection:
        connection.execute("DELETE FROM analyses")
    return jsonify({"status": "success", "message": "History cleared."})


@prediction_bp.get("/statistics")
def statistics():
    rows = history_rows(); total = len(rows); real = sum(row["prediction"] == "REAL" for row in rows)
    average = round(sum(row["confidence"] for row in rows) / total, 3) if total else 0
    return jsonify({"status": "success", "total": total, "real": real, "ai_generated": total - real, "average_confidence": average, "recent": rows[:5]})


@prediction_bp.get("/model-info")
def model_info():
    from model.model_loader import MODEL_PATH
    available = MODEL_PATH.is_file()
    payload = {"status": "success", "model": "Real vs AI-generated image classifier", "architecture": "StandardScaler + LogisticRegression", "framework": "scikit-learn", "classes": ["Real", "AI-Generated"], "input": "32 × 32 RGB + grayscale features", "model_available": available, "status_label": "Trained model active" if available else "Trained model artifact missing", "model_used": MODEL_NAME}
    try:
        metrics = json.loads((MODEL_PATH.parent / "metrics.json").read_text(encoding="utf-8"))
        payload["accuracy"] = round(metrics["accuracy"], 4)
        payload["test_samples"] = metrics.get("test_samples")
        matrix = metrics.get("confusion_matrix") or []
        if len(matrix) == 2 and sum(matrix[0]) and sum(matrix[1]):
            payload["recall_real"] = round(matrix[0][0] / sum(matrix[0]), 4)
            payload["recall_ai"] = round(matrix[1][1] / sum(matrix[1]), 4)
    except (OSError, ValueError, KeyError, TypeError):
        current_app.logger.warning("Could not read metrics.json for /api/model-info")
    return jsonify(payload)


@prediction_bp.get("/uploads/<path:filename>")
def uploaded_image(filename):
    return send_from_directory(current_app.config["UPLOAD_FOLDER"], filename)
