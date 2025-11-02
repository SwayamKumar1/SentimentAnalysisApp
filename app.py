import streamlit as st
import pickle
import re
import string
import nltk
import joblib
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

nltk.download('stopwords', quiet=True)

# Load the pre-trained model and vectorizer

model = joblib.load('sentiment_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'<.*?>', ' ', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\d+', ' ', text)
    text = text.strip()
    stop_words = set(stopwords.words('english'))
    text = re.sub(r'\s+', ' ', text)
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

# Streamlit Page Config
st.set_page_config(page_title="🎬 Sentiment Analyzer", page_icon="🧠", layout="centered")

# App Layout
st.markdown("""
    <style>
        .stApp {
            background-color: #0e1117;
            color: #fafafa;
            font-family: 'Segoe UI', sans-serif;
        }
        .main-title {
            text-align: center;
            font-size: 2.5rem;
            font-weight: bold;
            color: #00c0ff;
        }
        .subtitle {
            text-align: center;
            font-size: 1.1rem;
            color: #b3b3b3;
            margin-bottom: 30px;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown("<h1 class='main-title'>🎬 Sentiment Analysis App</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Analyze a movie review — Find out if it’s Positive or Negative!</p>", unsafe_allow_html=True)

st.title("Sentiment Analysis of Movie Reviews")
st.write("Enter a movie review below to analyze its sentiment.")

user_input = st.text_area("Enter the Movie Review:", "")

if st.button("Predict Sentiment"):
    if user_input.strip():
        cleaned_input = preprocess_text(user_input)
        input_vector = vectorizer.transform([cleaned_input])
        prediction = model.predict(input_vector)[0]
        proba = model.predict_proba(input_vector)[0]
        confidence = round(max(proba) * 100, 2) 
    
        if prediction.lower() == 'positive':
            st.success("🗣️ The review is POSITIVE!, You liked the movie")

        else:
            st.error("🗣️ The review is NEGATIVE!, You didn't like the movie")
        st.info(f"🤖 Model Confidence: {confidence}%")
    else:
        st.warning("Please enter a valid movie review!!! ")

st.markdown("---")
st.markdown("<p style='text-align:center; color:gray;'>Built with ❤️ using Streamlit & Scikit-learn</p>", unsafe_allow_html=True)
