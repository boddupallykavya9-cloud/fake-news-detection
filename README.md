# 📰 Fake News Detection System

An end-to-end Machine Learning web application that classifies news articles as **REAL** or **FAKE** using Natural Language Processing (NLP) techniques.

This project covers the complete ML lifecycle:
- Data preprocessing
- Feature engineering (TF-IDF)
- Model training & evaluation
- Model serialization
- Deployment using Flask API
- Live demo interface

---

## 🚀 Live Demo

🎥 Watch Demo Video:  
[Click here to watch](PASTE_YOUR_GOOGLE_DRIVE_LINK_HERE)

---

## 🛠️ Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- NLTK
- TF-IDF Vectorization
- Logistic Regression
- Flask (REST API)
- OBS (Demo recording)
- Git & GitHub

---

## 📂 Project Structure

```
fake-news-detection/
│
├── app.py
├── requirements.txt
├── README.md
│
├── models/
│   ├── fake_news_model.pkl
│   └── tfidf_vectorizer.pkl
│
└── notebooks/
    └── fake_news_analysis.ipynb

```
---

## 📊 Dataset

- **Dataset Used:** ISOT Fake News Dataset  
- Contains ~44,000 labeled news articles  
- Classes: FAKE (0) and REAL (1)

---

## 🧠 Methodology

1. Text cleaning & preprocessing
2. Stopword removal & normalization
3. TF-IDF feature extraction (Unigrams + Bigrams)
4. Model training using Logistic Regression
5. Evaluation using:
   - Accuracy
   - Precision
   - Recall
   - F1-score
6. Deployment using Flask API

---

## 📈 Model Performance

- Accuracy: ~99%
- Balanced precision & recall
- Evaluated using confusion matrix

---

## ▶️ How to Run Locally

### 1️⃣ Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/fake-news-detection.git
cd fake-news-detection
```
### 2️⃣ Install dependencies
```
pip install -r requirements.txt
```

### 3️⃣ Run the application
```
python app.py
```
Open in browser:
```
http://127.0.0.1:5000/test
```
### Example Test
### REAL:
```
The government approved a new renewable energy policy to reduce carbon emissions.
```

### FAKE:
```
Secret laboratory created invisible technology to control human minds.
```
### ⚠️ Limitations
- The model detects linguistic patterns, not factual correctness.
- Performance may vary for very short text inputs.

### 📌 Future Improvements
- Integrate transformer-based models (BERT)
- Deploy on cloud (Render / AWS)
- Add user authentication
- Add news source credibility scoring

### 👩‍💻 Author
Renuka
Boddupally Kavya

---

# 🔥 Important


PASTE_YOUR_GOOGLE_DRIVE_LINK_HERE
