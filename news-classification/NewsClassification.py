import streamlit as st # type: ignore
import joblib
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

# Download NLTK resources
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

# Load saved model and vectorizer
model = joblib.load("C:/Users/chaha/logistic_model.pkl")
vectorizer = joblib.load("C:/Users/chaha/tfidf_vectorizer.pkl")

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess(text):
    tokens = word_tokenize(text.lower())
    tokens = [lemmatizer.lemmatize(w) for w in tokens if w.isalpha() and w not in stop_words]
    return ' '.join(tokens)

# Streamlit UI
st.set_page_config(page_title="Fake News Classifier", page_icon="📰")
st.title("Fake News Detection App")
st.write("Enter the **title** and **body** of a news article:")

title_input = st.text_input("Title")
body_input = st.text_area("Body", height=200)

if st.button("Predict"):
    full_text = title_input + " " + body_input
    clean_text = preprocess(full_text)
    vec = vectorizer.transform([clean_text])
    pred = model.predict(vec)[0]
    proba = model.predict_proba(vec)[0]

    if pred == 1:
        st.success("This news article is likely **REAL**.")
    else:
        st.error("This news article is likely **FAKE**.")

    st.write(f"Confidence: {round(max(proba)*100, 2)}%")
