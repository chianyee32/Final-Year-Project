import os
import sys
import subprocess
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

# Initialize app, using the project root for static files
BASE_DIR = os.path.dirname(__file__)
app = Flask(__name__, static_folder=BASE_DIR, static_url_path="")
CORS(app, resources={r"/*": {"origins": "*"}})

# Folder to temporarily store uploads
UPLOAD_TEMP_DIR = os.path.join(BASE_DIR, "uploads_temp")
os.makedirs(UPLOAD_TEMP_DIR, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_TEMP_DIR

# Folder for storing prediction results (if needed)
PREDICTION_FOLDER = os.path.join(BASE_DIR, "uploads_prediction")
os.makedirs(PREDICTION_FOLDER, exist_ok=True)
app.config["PREDICTION_FOLDER"] = PREDICTION_FOLDER

@app.route('/')
def index():
    # Serve introduction.html directly from the static folder
    return app.send_static_file('introduction.html')

@app.route('/upload_prediction', methods=['POST'])
def upload_prediction():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    uploaded_file = request.files["file"]
    if uploaded_file.filename == "":
        return jsonify({"error": "No file selected"}), 400
    
    # Clear previous uploads
    for existing in os.listdir(app.config["UPLOAD_FOLDER"]):
        path = os.path.join(app.config["UPLOAD_FOLDER"], existing)
        if os.path.isfile(path):
            os.remove(path)

    # Save the new upload
    file_path = os.path.join(app.config["UPLOAD_FOLDER"], uploaded_file.filename)
    uploaded_file.save(file_path)

    try:
        # Run the prediction script with the same Python interpreter
        python_exe = sys.executable
        script_path = os.path.join(BASE_DIR, "predict_chemoresistance.py")

        result = subprocess.run(
            [python_exe, script_path, file_path],
            cwd=BASE_DIR,
            capture_output=True,
            text=True
        )

        # Log output
        print("✅ Prediction Output:", result.stdout)
        print("⚠️ Prediction Error (if any):", result.stderr)

        if result.returncode != 0:
            return jsonify({
                "error": "Prediction failed.",
                "details": result.stderr.strip()
            }), 400

        # Build download URL for the generated CSV
        output_filename = "predictions_output.csv"
        output_path = os.path.join(BASE_DIR, output_filename)

        if not os.path.exists(output_path):
            return jsonify({"error": "Prediction output file not found."}), 500

        return jsonify({
            "message": "Prediction completed successfully",
            "file_path": file_path,
            "output_file": output_filename,
            "download_url": f"{request.url_root}download_prediction/{output_filename}"
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/download_prediction/<filename>')
def download_prediction_file(filename):
    # Serve the generated file
    return send_from_directory(BASE_DIR, filename, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
