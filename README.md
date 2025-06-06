# 📰 Fake News Detection 

![Fake News Example](https://news.rpi.edu/sites/default/files/styles/sixteen_by_nine_medium/public/2024-03/FakeNewsWeb.jpg?itok=u5MuAtHK)

## 📌 Objective
The goal of this project is to develop a machine learning-based system that detects whether a given news article is **fake** or **real**. With the rise of misinformation, especially on social media platforms, this tool aims to help users and platforms make informed decisions about the credibility of online content.



---

## 🗂️ Project Structure

```
fake_news_detector/
├── notebooks/
│   └── fake_news_detector.ipynb     
│
├── models/
│   ├── naive_bayes_model.pkl
│   ├── logistic_regression_model.pkl
│   ├── random_forest_model.pkl
│   └── tfidf_vectorizer.pkl          
│
├── app/
│   └── app.py                        
│
├── requirements.txt                  
├── README.md                         
```

---

## 🚀 How to Run the Project

### 📥 Clone the Repository

```bash
git clone https://github.com/yourusername/fake_news_detector.git
cd fake_news_detector
```

### 🛠️ Set Up Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 🌐 Launch Streamlit App

```bash
cd app
streamlit run app.py
```

---

## 🌟 Features

- Binary classification of news: **Fake** or **Real**
- Exploratory Data Analysis (EDA) and visualization
- Multiple machine learning models for prediction:
  - Logistic Regression
  - Naive Bayes
  - Random Forest
- Interactive web interface using Streamlit
- TF-IDF based text feature engineering
- Pre-trained models stored and loaded for fast prediction

---

## 📊 Exploratory Data Analysis (EDA)

Explored:

- Class distribution (Fake vs. Real)
- Frequent terms using word clouds
- Text length distribution
- Common fake/real headlines


---

## 🧠 Model Details

| Model               | Description                             |
|--------------------|-----------------------------------------|
| **Naive Bayes**     | Probabilistic model ideal for text classification |
| **Logistic Regression** | Linear model with good generalization performance |
| **Random Forest**   | Ensemble of decision trees for better accuracy |

**Feature Extraction**:  
- Used **TF-IDF vectorizer** for converting text into numerical vectors.

**Evaluation Metrics**:
- Accuracy
- F1-Score
- Confusion Matrix

---


## 🖥️ Streamlit Web App UI

The Streamlit app provides:

- A simple input text box to paste news articles
- Instant prediction output (Real or Fake)
- Clean and interactive user interface built with Streamlit

---

## 📦 Requirements

Key libraries (see `requirements.txt`):

- `scikit-learn`
- `pandas`
- `numpy`
- `streamlit`
- `matplotlib`, `seaborn`
- `nltk`, `textblob` 

---

## 🔮 Future Enhancements

- Deep learning models (BERT, LSTM)
- Real-time news scraping and prediction
- Multilingual support
- Explainability with SHAP or LIME
- Deployment on cloud (Heroku, AWS, etc.)

---

## 🙌 Acknowledgments

- Dataset from Huggingface
- Libraries: scikit-learn, Streamlit, NLTK, TextBlob
- Inspiration from real-world fake news detection challenges

---

## 📬 Contact

**Tanzila Parvez Akhtar**  
🔗 [LinkedIn](https://linkedin.com/in/tanzila-pervaiz)
