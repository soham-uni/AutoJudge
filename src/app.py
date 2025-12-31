import streamlit as st
import joblib
import re
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from scipy.sparse import hstack

# 1. Page Configuration
st.set_page_config(page_title="AutoJudge AI", page_icon="⚖️", layout="wide")

# 2. Resource Loading
@st.cache_resource
def load_assets():
    clf = joblib.load('models/classifier.pkl')
    reg = joblib.load('models/regressor.pkl')
    tfidf = joblib.load('models/vectorizer.pkl')
    return clf, reg, tfidf

try:
    clf, reg, tfidf = load_assets()
except Exception:
    st.error("Models not found! Please run train.py first.")
    st.stop()

# 3. Preprocessing Function
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s\^\*]', ' ', text)
    words = text.split()
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    cleaned = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]
    return " ".join(cleaned)

# Training-consistent constraint feature
def get_complexity_signal(text):
    matches = re.findall(r'(?:10(?:\^|\*\*|e)|1000)\s*(\d+)', text)
    magnitudes = [int(m) for m in matches if m.isdigit()]
    return max(magnitudes) if magnitudes else 0

# 4. Styled UI Header
st.markdown("""
    <style>
    .stTextArea textarea { font-size: 16px; }
    .prediction-card { 
        background-color: #1e2130; 
        padding: 25px; 
        border-radius: 15px; 
        border-left: 5px solid #ff4b4b;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

st.title("AutoJudge: CP Difficulty Predictor")
st.write("Paste your competitive programming problem details below to predict its difficulty level and rating.")

col1, col2 = st.columns([2, 1])

with col1:
    desc = st.text_area("📝 Problem Description", height=200)
    inp_desc = st.text_area("📥 Input Description", height=100)
    out_desc = st.text_area("📤 Output Description", height=100)

with col2:
    st.subheader("Analysis Center")
    if st.button("Predict Difficulty"):
        if not desc or not inp_desc:
            st.warning("Please provide problem details.")
        else:
            combined = f"{desc} {inp_desc} {out_desc}"
            cleaned = clean_text(combined)

            # === Feature Extraction (MATCHES TRAINING EXACTLY) ===
            tfidf_feat = tfidf.transform([cleaned])

            text_len = len(cleaned)
            math_symbols = len(re.findall(r'[+\-*/%=<>!^]', combined))
            max_constraint = get_complexity_signal(combined)

            keywords = ['graph', 'dp', 'tree', 'segment', 'dijkstra', 'shortest', 'query', 
                        'array', 'string', 'recursion', 'complexity', 'optimal', 'greedy',
                        'bitwise', 'modulo', 'combinatorics', 'probability', 'geometry']

            kw_feats = [1 if k in cleaned else 0 for k in keywords]

            custom_feats = np.array([[text_len, math_symbols, max_constraint] + kw_feats])

            X_input = hstack([tfidf_feat, custom_feats])

            # 5. Prediction Logic
            class_idx = clf.predict(X_input)[0]
            score_pred = reg.predict(X_input)[0]

            labels = {0: "Easy", 1: "Medium", 2: "Hard"}

            st.markdown(f"""
                <div class="prediction-card">
                    <p style="margin:0; color: #888;">PREDICTED CLASS</p>
                    <h1 style="color: #ff4b4b; margin-bottom: 10px;">{labels[class_idx]}</h1>
                    <p style="margin:0; color: #888;">DIFFICULTY RATING</p>
                    <h2 style="color: #4bafff;">{score_pred:.2f}</h2>
                </div>
            """, unsafe_allow_html=True)

            st.progress(min(max(float(score_pred) / 10.0, 0.0), 1.0))
