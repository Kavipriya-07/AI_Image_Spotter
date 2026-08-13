"""Flask application entry point for AI Image Spotter."""
from flask import Flask, jsonify, render_template, send_from_directory
from werkzeug.exceptions import RequestEntityTooLarge

from backend.routes.prediction import prediction_bp


def create_app():
    app = Flask(__name__)
    app.config.update(
        MAX_CONTENT_LENGTH=10 * 1024 * 1024,
        UPLOAD_FOLDER="uploads",
        SECRET_KEY="change-this-before-production",
    )
    app.register_blueprint(prediction_bp)

    @app.get("/")
    def index():
        return render_template("index.html")

    @app.get("/uploads/<path:filename>")
    def uploaded_image(filename):
        return send_from_directory(app.config["UPLOAD_FOLDER"], filename)

    @app.errorhandler(RequestEntityTooLarge)
    def file_too_large(_error):
        return jsonify({"status": "error", "message": "Image must be 10 MB or smaller."}), 413

    @app.errorhandler(404)
    def not_found(_error):
        return jsonify({"status": "error", "message": "Endpoint not found."}), 404

    return app


if __name__ == "__main__":
    create_app().run(debug=True)
