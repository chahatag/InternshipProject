# Fake News Classification (Real vs Fake)

## Objective
Classify news articles as **real or fake** using Natural Language Processing (NLP) and supervised machine learning.

## Dataset
- Collected from Kaggle: [`Fake.csv`] + [`True.csv`]
- Columns: `Title`, `Text`, `Subject`, `Date`

## Tools & Libraries
- Python
- Pandas, Numpy
- Scikit-learn (TF-IDF, Logistic Regression, Naive Bayes)
- NLTK (for text preprocessing)
- Streamlit (for web demo)

## Project Workflow
1. **Merge & label** real and fake datasets.
2. **Clean & preprocess text** (removing stopwords, punctuation, etc.).
3. **Vectorize** text using TF-IDF.
4. **Train & evaluate** Logistic Regression / Naive Bayes.
5. **Build Streamlit app** to take input news and predict if it's Real or Fake.

## Evaluation Metrics
- Accuracy
- F1 Score
- Precision
- Recall

## Web Demo 
Use the `NewsClassification.py` file in Streamlit to run:
```bash
streamlit run NewsClassification.py
