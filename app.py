"""
Smart Image Recognition System
================================
Flask application entry point.
Handles routing, file upload, and prediction pipeline.
"""

import os
import uuid
import logging
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from model.classifier import ImageClassifier

# ─── App Configuration ────────────────────────────────────────────────────────

app = Flask(__name__)

# Secret key for session security
app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "dev-secret-change-in-prod")

# Upload settings
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), "uploads")
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp", "bmp", "gif"}
MAX_CONTENT_LENGTH = 10 * 1024 * 1024  # 10 MB limit

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH

# Ensure upload directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ─── Logging ──────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s"
)
logger = logging.getLogger(__name__)

# ─── Load Model (once at startup) ─────────────────────────────────────────────

logger.info("Loading image classifier model…")
classifier = ImageClassifier()
logger.info("Model ready.")

# ─── Helpers ──────────────────────────────────────────────────────────────────

def allowed_file(filename: str) -> bool:
    """Check that the file has an allowed extension."""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def unique_filename(filename: str) -> str:
    """Generate a UUID-prefixed filename to prevent collisions and path traversal."""
    ext = filename.rsplit(".", 1)[1].lower()
    return f"{uuid.uuid4().hex}.{ext}"

# ─── Routes ───────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    """Serve the main upload & results page."""
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    """
    POST /predict
    Accepts a multipart image upload, runs inference, returns JSON:
    {
        "success": true,
        "filename": "abc123.jpg",
        "predictions": [
            {"label": "tabby cat", "confidence": 94.2},
            ...
        ]
    }
    """
    # ── Validate request ──────────────────────────────────────────────────────
    if "image" not in request.files:
        return jsonify({"success": False, "error": "No image field in request."}), 400

    file = request.files["image"]

    if file.filename == "":
        return jsonify({"success": False, "error": "No file selected."}), 400

    if not allowed_file(file.filename):
        return jsonify({
            "success": False,
            "error": f"File type not allowed. Accepted: {', '.join(ALLOWED_EXTENSIONS)}"
        }), 415

    # ── Save to disk ──────────────────────────────────────────────────────────
    safe_name = unique_filename(secure_filename(file.filename))
    save_path = os.path.join(app.config["UPLOAD_FOLDER"], safe_name)
    file.save(save_path)
    logger.info("Saved upload: %s", save_path)

    # ── Run inference ─────────────────────────────────────────────────────────
    try:
        predictions = classifier.predict(save_path, top_k=3)
    except Exception as exc:
        logger.exception("Prediction failed: %s", exc)
        return jsonify({"success": False, "error": "Prediction failed. Please try another image."}), 500

    # ── Return result ─────────────────────────────────────────────────────────
    return jsonify({
        "success": True,
        "filename": safe_name,
        "predictions": predictions
    })


@app.errorhandler(413)
def too_large(_):
    return jsonify({"success": False, "error": "File too large. Maximum size is 10 MB."}), 413


@app.errorhandler(404)
def not_found(_):
    return render_template("index.html"), 404

# ─── Entry Point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_ENV", "production") == "development"
    app.run(host="0.0.0.0", port=port, debug=debug)