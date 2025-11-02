# 🎬 Sentiment Analysis App

An end-to-end **Sentiment Analysis** web app built with **Python**, **Scikit-Learn**, and **Streamlit**, capable of classifying IMDB movie reviews as *Positive* or *Negative* with over **89% accuracy**.

---

## 🚀 Features

- 🧹 **Text Preprocessing** — cleans HTML tags, punctuation, numbers, and stopwords  
- 🧠 **TF-IDF Vectorization** — converts text into numerical features for the model  
- 🤖 **Logistic Regression Model** — trained on the IMDB 50K review dataset  
- 🌐 **Interactive Streamlit UI** — type or paste any review to see instant prediction  
- 💬 **Confidence Score** — displays model confidence for every prediction  

---

## 📂 Project Structure
SentimentAnalysisApp/
│
├── app.py # Streamlit web app
├── train_model.py # Model training script
├── sentiment_model.pkl # Trained Logistic Regression model
├── tfidf_vectorizer.pkl # Saved TF-IDF vectorizer
├── IMDB Dataset.csv # Dataset (50,000 labeled movie reviews)
├── requirements.txt # Dependencies
└── README.md # Project documentation

yaml
Copy code

---

## ⚙️ Installation & Usage

### 1️⃣ Clone the repository
```bash
git clone https://github.com/SwayamKumar1/SentimentAnalysisApp.git
cd SentimentAnalysisApp
2️⃣ (Optional) Create a virtual environment

python -m venv venv
venv\Scripts\activate  # Windows
# OR
source venv/bin/activate  # macOS/Linux

#3️⃣ Install dependencies
pip install -r requirements.txt

#4️⃣ Run the Streamlit app
streamlit run app.py
Then open the local URL shown in your terminal (usually http://localhost:8501).

#🧩 Model Performance
Metric	Score
Accuracy	0.895
Precision	0.90
Recall	0.89
F1-Score	0.90

```
## 🧠 How It Works

- The model is trained on IMDB movie reviews (50,000 entries).
- Each review is cleaned and transformed into TF-IDF vectors.
- Logistic Regression predicts whether the sentiment is positive or negative.
- Streamlit provides an interactive UI to test new reviews.

## 👨‍💻 Author
Swayam Kumar
Data Science & AI Student | Machine Learning Enthusiast
🔗 GitHub

## 🏁 Future Improvements
- Add LSTM/Transformer-based model for better accuracy
- Integrate API endpoints for scalable deployment
- Add dashboard visualization for review trends

⭐ If you like this project, give it a star on GitHub!
