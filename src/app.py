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
    except Exception as e:
        pass

download_nltk_data()

# --- Page Configuration ---
st.set_page_config(page_title="AutoJudge AI", page_icon="⚖️", layout="centered")

# --- Custom Styling (Fancy CSS) ---
st.markdown("""
    <style>
    /* Main Background */
    .stApp {
        background: radial-gradient(circle at 50% 50%, #1a1c2c 0%, #0a0b10 100%);
    }

    /* Glassmorphism Card */
    .glass-card {
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 20px;
        padding: 30px;
        margin-bottom: 20px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
    }

    /* Glowing Title */
    .main-title {
        background: linear-gradient(90deg, #ff4b4b, #ff9b4b);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3.5rem;
        font-weight: 800;
        text-align: center;
        margin-bottom: 0px;
        letter-spacing: -2px;
    }

    /* Prediction Result Card */
    .prediction-container {
        text-align: center;
        padding: 40px;
        border-radius: 25px;
        background: linear-gradient(135deg, rgba(255, 75, 75, 0.1) 0%, rgba(75, 175, 255, 0.1) 100%);
        border: 1px solid rgba(255, 255, 255, 0.15);
        margin-top: 30px;
        animation: fadeIn 0.8s ease-in-out;
    }

    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }

    /* Custom Input Box */
    .stTextArea textarea {
        background-color: rgba(0, 0, 0, 0.2) !important;
        color: #e0e0e0 !important;
        border: 1px solid #3d4156 !important;
        border-radius: 12px !important;
    }

    /* Fancy Button */
    .stButton>button {
        width: 100%;
        background: linear-gradient(90deg, #ff4b4b, #ff7b4b) !important;
        color: white !important;
        font-weight: bold !important;
        border: none !important;
        padding: 15px !important;
        border-radius: 12px !important;
        text-transform: uppercase;
        letter-spacing: 2px;
        transition: 0.3s all ease;
    }
    
    .stButton>button:hover {
        transform: scale(1.02);
        box-shadow: 0 0 20px rgba(255, 75, 75, 0.4);
    }
    </style>
""", unsafe_allow_html=True)

# --- Asset Loading ---
@st.cache_resource
def load_assets():
    try:
        clf = joblib.load('models/classifier.pkl')
        reg = joblib.load('models/regressor.pkl')
        tfidf = joblib.load('models/vectorizer.pkl')
        return clf, reg, tfidf
    except:
        return None, None, None

clf, reg, tfidf = load_assets()

# --- Logic Functions ---
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s\^\*]', ' ', text)
    words = text.split()
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    cleaned = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]
    return " ".join(cleaned)

def get_complexity_signal(text):
    matches = re.findall(r'(?:10(?:\^|\*\*|e)|1000)\s*(\d+)', text)
    magnitudes = [int(m) for m in matches if m.isdigit()]
    return max(magnitudes) if magnitudes else 0

# --- UI Components ---
st.markdown('<h1 class="main-title">AutoJudge AI</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align:center; color:#888; margin-bottom:40px;">Competitive Programming Problem Difficulty Oracle</p>', unsafe_allow_html=True)

# Main Form Container
with st.container():
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    
    full_problem = st.text_area(
        "📝 Paste Full Problem Description", 
        placeholder="Paste everything: statement, input, and output constraints here...",
        height=350,
        help="Our AI will automatically parse the constraints and context."
    )
    
    st.markdown("<br>", unsafe_allow_html=True)
    predict_btn = st.button("Analyze Complexity")
    
    st.markdown('</div>', unsafe_allow_html=True)

# --- Prediction Engine ---
if predict_btn:
    if not full_problem or len(full_problem) < 20:
        st.warning("Please provide a more detailed problem description.")
    elif clf is None:
        st.error("Error: Model files not found in /models directory.")
    else:
        with st.status("🔍 Analyzing problem context...", expanded=True) as status:
            cleaned = clean_text(full_problem)
            st.write("Extracting TF-IDF features...")
            tfidf_feat = tfidf.transform([cleaned])
            
            st.write("Calculating complexity signals...")
            text_len = len(cleaned)
            math_symbols = len(re.findall(r'[+\-*/%=<>!^]', full_problem))
            max_constraint = get_complexity_signal(full_problem)
            
            keywords = ['graph', 'dp', 'tree', 'segment', 'dijkstra', 'shortest', 'query', 
                        'array', 'string', 'recursion', 'complexity', 'optimal', 'greedy',
                        'bitwise', 'modulo', 'combinatorics', 'probability', 'geometry']
            kw_feats = [1 if k in cleaned else 0 for k in keywords]
            
            custom_feats = np.array([[text_len, math_symbols, max_constraint] + kw_feats])
            X_input = hstack([tfidf_feat, custom_feats])
            
            time.sleep(0.5)
            status.update(label="Analysis Complete!", state="complete", expanded=False)

        # Predictions
        class_idx = clf.predict(X_input)[0]
        score_pred = reg.predict(X_input)[0]
        labels = {0: "Easy (Greedy/Math)", 1: "Medium (DP/Data Structures)", 2: "Hard (Advanced Algo)"}
        accent_color = "#ff4b4b" if class_idx == 2 else ("#ff9b4b" if class_idx == 1 else "#4bffab")

        # Results Display
        st.markdown(f"""
            <div class="prediction-container">
                <p style="color:#aaa; font-size: 0.9rem; text-transform: uppercase; letter-spacing: 2px;">Predicted Level</p>
                <h1 style="color:{accent_color}; font-size: 3.5rem; margin: 0;">{labels[class_idx].split(' ')[0]}</h1>
                <p style="color:#666; font-style: italic; margin-bottom: 20px;">{labels[class_idx].split('(')[1].replace(')', '')}</p>
                <hr style="opacity:0.1">
                <p style="color:#aaa; margin-top: 20px; font-size: 0.9rem;">DIFFICULTY SCORE</p>
                <h2 style="color:#4bafff; font-family: 'Courier New', monospace; font-size: 2.5rem; margin-top:0;">{score_pred:.2f}</h2>
            </div>
        """, unsafe_allow_html=True)
        
        # Progress Bar as a "Meter"
        # Assuming difficulty scale 0 to 10
        meter_val = min(max(float(score_pred) / 10.0, 0.0), 1.0)
        st.markdown(f"<p style='text-align:center; color:#555; margin-top:10px;'>Complexity Intensity</p>", unsafe_allow_html=True)
        st.progress(meter_val)

# Footer
st.markdown("<br><p style='text-align:center; color:#333; font-size:0.8rem;'>AutoJudge AI Engine v2.0 • Powered by Scikit-Learn</p>", unsafe_allow_html=True)