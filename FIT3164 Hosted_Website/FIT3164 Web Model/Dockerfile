# syntax=docker/dockerfile:1
FROM python:3.10-slim

# Install system deps (if needed for RDKit, you might need apt packages here)
# RUN apt-get update && apt-get install -y --no-install-recommends \
#     build-essential \
#     && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your code AND your Preprocess_files, model, and scaler
COPY . .

# Expose the port your Flask app listens on
EXPOSE 5000

# Start the app with Gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "connection:app"]
