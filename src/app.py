import streamlit as st
import joblib
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk

# Load resources
nltk.download('stopwords')
nltk.download('wordnet')

# Load models and vectorizer
clf = joblib.load('models/classifier.pkl')
reg = joblib.load('models/regressor.pkl')
tfidf = joblib.load('models/vectorizer.pkl')

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
    words = text.split()
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    cleaned = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]
    return " ".join(cleaned)

st.title("🚀 AutoJudge: CP Problem Difficulty Predictor")
st.markdown("Enter problem details below to predict its difficulty.")

# 1. Text Inputs 
desc = st.text_area("Problem Description")
inp_desc = st.text_area("Input Description")
out_desc = st.text_area("Output Description")

if st.button("Predict Difficulty"): 
    # 2. Preprocess input
    combined = f"{desc} {inp_desc} {out_desc}"
    cleaned = clean_text(combined)
    
    # 3. Feature Extraction 
    # TF-IDF
    tfidf_feat = tfidf.transform([cleaned])
    
    # Manual Features 
    text_len = len(cleaned)
    math_symbols = len(re.findall(r'[+\-*/%=<>!^]', combined))
    
    # Keyword flags 
    keywords = ['graph', 'dp', 'tree', 'segment', 'dijkstra', 'shortest', 'query', 'array', 'string', 'recursion']
    kw_feats = [1 if k in cleaned else 0 for k in keywords]
    
    # Stack features
    import numpy as np
    from scipy.sparse import hstack
    custom_feats = np.array([[text_len, math_symbols] + kw_feats])
    X_input = hstack([tfidf_feat, custom_feats])
    
    # 4. Make Predictions 
    class_idx = clf.predict(X_input)[0]
    score_pred = reg.predict(X_input)[0]
    
    # Map back to labels
    labels = {0: "Easy", 1: "Medium", 2: "Hard"}
    
    # 5. Display Results
    st.subheader(f"Predicted Class: {labels[class_idx]}")
    st.subheader(f"Predicted Score: {score_pred:.2f}")