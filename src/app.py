import streamlit as st
import joblib
import re
import numpy as np
import nltk
import time
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from scipy.sparse import hstack

# --- Resource Initialization ---
@st.cache_resource
def download_nltk_data():
    try:
        nltk.download('stopwords', quiet=True)
        nltk.download('wordnet', quiet=True)
        nltk.download('omw-1.4', quiet=True)
    except:
        pass

download_nltk_data()

# --- Page Configuration ---
st.set_page_config(page_title="AutoJudge AI", page_icon="⚖️", layout="centered")

# --- Custom Styling ---
st.markdown("""
<style>
.stApp { background: radial-gradient(circle at 50% 50%, #1a1c2c 0%, #0a0b10 100%); }

.glass-card {
    background: rgba(255,255,255,0.03);
    backdrop-filter: blur(10px);
    border-radius: 20px;
    padding: 30px;
    border: 1px solid rgba(255,255,255,0.1);
    box-shadow: 0 8px 32px rgba(0,0,0,0.37);
}

.main-title {
    background: linear-gradient(90deg,#ff4b4b,#ff9b4b);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 3.5rem;
    font-weight: 800;
    text-align: center;
}

.prediction-container {
    text-align: center;
    padding: 40px;
    border-radius: 25px;
    background: linear-gradient(135deg,rgba(255,75,75,0.1),rgba(75,175,255,0.1));
    border: 1px solid rgba(255,255,255,0.15);
}

.stTextArea textarea {
    background-color: rgba(0,0,0,0.2) !important;
    color: #e0e0e0 !important;
    border-radius: 12px !important;
}

.stButton>button {
    width: 100%;
    background: linear-gradient(90deg,#ff4b4b,#ff7b4b) !important;
    color: white !important;
    font-weight: bold !important;
    border-radius: 12px !important;
}
</style>
""", unsafe_allow_html=True)

# --- Load Models ---
@st.cache_resource
def load_assets():
    return (
        joblib.load("models/classifier.pkl"),
        joblib.load("models/regressor.pkl"),
        joblib.load("models/vectorizer.pkl")
    )

clf, reg, tfidf = load_assets()

# --- Helpers ---
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s\^\*]', ' ', text)
    words = text.split()
    stop_words = set(stopwords.words('english'))
    lem = WordNetLemmatizer()
    return " ".join(lem.lemmatize(w) for w in words if w not in stop_words)

def get_complexity_signal(text):
    matches = re.findall(r'(?:10(?:\^|\*\*|e)|1000)\s*(\d+)', text)
    return max([int(x) for x in matches], default=0)

# --- UI ---
st.markdown('<h1 class="main-title">AutoJudge AI</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align:center;color:#888;">Competitive Programming Difficulty Oracle</p>', unsafe_allow_html=True)

with st.container():
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)

    prob_desc = st.text_area("🧩 Problem Description", height=200)
    input_desc = st.text_area("📥 Input Description", height=150)
    output_desc = st.text_area("📤 Output Description", height=150)

    predict_btn = st.button("Analyze Complexity")
    st.markdown('</div>', unsafe_allow_html=True)

# --- Prediction ---
if predict_btn and prob_desc and input_desc and output_desc:
    combined_text = f"{prob_desc} {input_desc} {output_desc}"
    cleaned = clean_text(combined_text)
    tfidf_feat = tfidf.transform([cleaned])

    text_len = len(cleaned)
    math_symbols = len(re.findall(r'[+\-*/%=<>!^]', combined_text))
    max_constraint = get_complexity_signal(combined_text)

    keywords = ['graph','dp','tree','segment','dijkstra','shortest','query','array',
                'string','recursion','complexity','optimal','greedy','bitwise','modulo',
                'combinatorics','probability','geometry']
    kw_feats = [1 if k in cleaned else 0 for k in keywords]

    X = hstack([tfidf_feat, np.array([[text_len, math_symbols, max_constraint] + kw_feats])])

    class_idx = clf.predict(X)[0]
    score = float(reg.predict(X)[0])

    labels = {0:"Easy",1:"Medium",2:"Hard"}
    color = "#4bffab" if class_idx==0 else "#ff9b4b" if class_idx==1 else "#ff4b4b"

    st.markdown(f"""
    <div class="prediction-container">
        <p style="color:#aaa;">Predicted Level</p>
        <h1 style="color:{color};">{labels[class_idx]}</h1>
        <p style="color:#aaa;">Difficulty Score</p>
        <h2 style="color:#4bafff;">{score:.2f}</h2>
    </div>
    """, unsafe_allow_html=True)

    st.progress(min(score/10, 1.0))
