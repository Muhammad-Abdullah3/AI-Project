from flask import Flask, render_template, request, jsonify
import requests
import pandas as pd
import os

from datetime import datetime

app = Flask(__name__)

@app.context_processor
def inject_now():
    return {'datetime_now': datetime.now().strftime("%d %B %Y | %H:%M")}

API_BASE_URL = "http://localhost:8000"

@app.route('/')
def dashboard():
    try:
        # Get filters from query string
        params = {
            "province": request.args.get('province'),
            "gender": request.args.get('gender'),
            "performance": request.args.get('performance'),
            "income": request.args.get('income')
        }
        # Filter out None values
        params = {k: v for k, v in params.items() if v is not None}
        
        response = requests.get(f"{API_BASE_URL}/analytics/summary", params=params)
        summary_data = response.json() if response.status_code == 200 else None
        return render_template('dashboard.html', summary=summary_data)
    except Exception as e:
        print(f"Error fetching summary: {e}")
        return render_template('dashboard.html', summary=None)

@app.route('/advisor')
def diagnostic():
    return render_template('diagnostic.html')

@app.route('/batch')
def batch():
    return render_template('batch.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    try:
        response = requests.post(f"{API_BASE_URL}/predict/student", json=data)
        return jsonify(response.json()), response.status_code
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/upload/batch', methods=['POST'])
def upload_batch():
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400
    
    file = request.files['file']
    try:
        files = {'file': (file.filename, file.stream, file.content_type)}
        response = requests.post(f"{API_BASE_URL}/upload/class", files=files)
        return jsonify(response.json()), response.status_code
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(port=5000, debug=True)
