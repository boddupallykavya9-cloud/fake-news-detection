import sys
import re
from pathlib import Path
from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib

# --------------------------------------------------
# Project Setup
# --------------------------------------------------

project_root = Path(__file__).parent.absolute()
sys.path.insert(0, str(project_root))

print(f"📂 Project root: {project_root}")

# --------------------------------------------------
# Text Preprocessing (MUST MATCH TRAINING LOGIC)
# --------------------------------------------------

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)           # Remove URLs
    text = re.sub(r"[^a-zA-Z\s]", "", text)       # Remove punctuation & numbers
    text = re.sub(r"\s+", " ", text)              # Remove extra spaces
    return text.strip()

# --------------------------------------------------
# Load Model & Vectorizer
# --------------------------------------------------

model = None
vectorizer = None

try:
    model = joblib.load("fake_news_model.pkl")
    vectorizer = joblib.load("tfidf_vectorizer.pkl")
    print("✅ Model and vectorizer loaded successfully.")
except Exception as e:
    print(f"❌ Error loading models: {e}")

# --------------------------------------------------
# Flask App Setup
# --------------------------------------------------

app = Flask(__name__)
CORS(app)  # Enable CORS for browser testing

# --------------------------------------------------
# Routes
# --------------------------------------------------

@app.route("/")
def home():
    return """
    <h1>🚀 Fake News Detection API</h1>
    <p>Server is running successfully.</p>
    <p>Use POST /predict to get predictions.</p>
    <p>Or visit <a href="/test">/test</a> for quick testing.</p>
    """

@app.route("/health")
def health():
    return jsonify({
        "status": "running",
        "model_loaded": model is not None,
        "vectorizer_loaded": vectorizer is not None
    })

@app.route("/test")
def test_form():
    return """
    <h2>📰 Quick Test</h2>
    <form action="/predict" method="post">
        <textarea name="text" rows="6" cols="60"
        placeholder="Enter news text here..."></textarea><br><br>
        <input type="submit" value="Analyze News"
        style="padding:10px 20px;font-size:16px;">
    </form>
    """

@app.route("/predict", methods=["POST"])
def predict():
    if model is None or vectorizer is None:
        return jsonify({"error": "Model not loaded properly."}), 500

    try:
        # Get text from JSON or form
        if request.is_json:
            data = request.get_json()
            text = data.get("text", "")
        else:
            text = request.form.get("text", "")

        if not text.strip():
            return jsonify({"error": "No text provided"}), 400

        # Preprocess text
        cleaned_text = clean_text(text)

        # Vectorize
        vector = vectorizer.transform([cleaned_text])

        # Predict
        prediction = model.predict(vector)[0]

        # Confidence (Logistic Regression supports predict_proba)
        if hasattr(model, "predict_proba"):
            confidence = float(model.predict_proba(vector).max() * 100)
        else:
            confidence = None

        result = "REAL" if prediction == 1 else "FAKE"

        response = {
            "prediction": result,
            "confidence": f"{confidence:.2f}%" if confidence else "N/A",
            "input_preview": text[:120] + "..." if len(text) > 120 else text
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

# --------------------------------------------------
# Run Server
# --------------------------------------------------

if __name__ == "__main__":
    print("🚀 Starting Fake News Detection API...")
    print("🌐 http://127.0.0.1:5000")
    app.run(debug=True, host="127.0.0.1", port=5000)