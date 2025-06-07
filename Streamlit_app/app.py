import streamlit as st
import joblib
import os

# Get absolute path to the models folder relative to this script
BASE_DIR = os.path.dirname(__file__)
MODELS_DIR = os.path.join(BASE_DIR, '..', 'Models')

# Load models and vectorizer with correct paths
lr_model = joblib.load(os.path.join(MODELS_DIR, 'lr_model.pkl'))
rf_model = joblib.load(os.path.join(MODELS_DIR, 'rf_model.pkl'))
nb_model = joblib.load(os.path.join(MODELS_DIR, 'nb_model.pkl'))
tfidf_vectorizer = joblib.load(os.path.join(MODELS_DIR, 'tfidf_vectorizer.pkl'))

def data_cleaning(text):
    import re
    import string
    text = text.lower()
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
    return text

def predict_news(text):
    cleaned_text = data_cleaning(text)
    vectorized_text = tfidf_vectorizer.transform([cleaned_text])
    
    label_map = {0: "Fake", 1: "Real"}
    
    predictions = {
        "Logistic Regression": label_map[lr_model.predict(vectorized_text)[0]],
        "Random Forest": label_map[rf_model.predict(vectorized_text)[0]],
        "Naive Bayes": label_map[nb_model.predict(vectorized_text)[0]]
    }
    
    return predictions

# Streamlit UI
st.set_page_config(page_title="News Authenticity Checker", layout="centered")
st.title("📰 Fake News Detection App")
st.write("Enter a news headline or article snippet to check if it's real or fake.")

user_input = st.text_area("Enter news text here:")

if st.button("Predict"):
    if not user_input.strip():
        st.warning("Please enter some text.")
    else:
        results = predict_news(user_input)
        st.subheader("Predictions:")
        for model_name, prediction in results.items():
            if prediction == "Real":
                st.success(f"{model_name}: This news is **Real**.")
            else:
                st.error(f"{model_name}: This news is **Fake**.")
