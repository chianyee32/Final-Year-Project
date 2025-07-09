# File: connection.py

import os
import sys
import subprocess
import re
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

# ─── Force UTF-8 on Windows to avoid encoding errors ───────────────────────────
if sys.platform.startswith("win"):
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# Setup directories
BASE_DIR          = os.path.dirname(__file__)
UPLOAD_FOLDER     = os.path.join(BASE_DIR, "uploads_temp")
PREDICTION_FOLDER = os.path.join(BASE_DIR, "uploads_prediction")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PREDICTION_FOLDER, exist_ok=True)

# Initialize Flask
app = Flask(__name__, static_folder=BASE_DIR, static_url_path="")
CORS(app)

@app.route("/", methods=["GET"])
def index():
    return app.send_static_file("introduction.html")

@app.route("/upload_prediction", methods=["POST"])
def upload_prediction():
    file = request.files.get("file")
    if not file or not file.filename:
        return jsonify(error="No file uploaded"), 400

    # Save upload
    in_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(in_path)
    print(f"[UPLOAD] Saved file: {in_path}", file=sys.stderr)

    # Run prediction script
    proc = subprocess.run(
        [sys.executable, os.path.join(BASE_DIR, "predict_chemoresistance.py"), in_path],
        capture_output=True,
        text=True
    )

    # Filter out TensorFlow and time-stamp lines from stderr
    raw_err = proc.stderr or ""
    filtered = [l for l in raw_err.splitlines()
                if l.strip() and
                   not re.match(r"^\d{4}-\d{2}-\d{2}", l) and
                   "tensorflow" not in l.lower()]
    user_error = filtered[-1].strip() if filtered else raw_err.strip()

    # Log for debugging
    out = proc.stdout.strip()
    if out:
        print("[PREDICT STDOUT]", out, file=sys.stderr)
    if user_error:
        print("[PREDICT STDERR]", user_error, file=sys.stderr)

    # On failure, only return the cleaned user_error
    if proc.returncode != 0:
        return jsonify(error="Prediction failed", details=user_error), 400

    # Move the predictions file
    src = os.path.join(BASE_DIR, "predictions_output.csv")
    dst = os.path.join(PREDICTION_FOLDER, "predictions_output.csv")
    if not os.path.exists(src):
        return jsonify(error="Output missing"), 500
    os.replace(src, dst)
    print(f"[OUTPUT] Moved to {dst}", file=sys.stderr)

    # Success
    return jsonify(
        message="Prediction succeeded",
        download_url="/download_prediction/predictions_output.csv"
    )

@app.route("/download_prediction/<name>", methods=["GET"])
def download_prediction(name):
    return send_from_directory(PREDICTION_FOLDER, name, as_attachment=True)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5002))
    app.run(host="0.0.0.0", port=port, debug=True)
