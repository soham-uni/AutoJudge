import pandas as pd
import json
import re
import nltk
import joblib
import math
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from scipy.sparse import hstack

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

def clean_problem_text(text):
    text = text.lower()
    text = re.sub(r'<.*?>', ' ', text)
    text = re.sub(r'[^a-z0-9\s\^\*\+\-<=]', ' ', text)
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    return " ".join(
        lemmatizer.lemmatize(w)
        for w in text.split()
        if w not in stop_words
    )

def extract_constraint_log(text):
    patterns = [
        r'(\d+)\s*\*\s*10\^(\d+)',
        r'(\d+)e(\d+)',
        r'10\^(\d+)',
        r'<=\s*(\d+)',
        r'up to\s*(\d+)'
    ]

    vals = []
    for p in patterns:
        for m in re.findall(p, text.lower()):
            try:
                if isinstance(m, tuple):
                    if len(m) == 2:
                        v = int(m[0]) * (10 ** int(m[1]))
                else:
                    v = 10 ** int(m)

                if v > 0:
                    vals.append(math.log10(v))
            except:
                pass

    return max(vals) if vals else 0.0

def extract_features(df):
    df['text_len'] = df['cleaned_text'].str.len()
    df['math_ops'] = df['combined_text'].apply(
        lambda x: len(re.findall(r'[+\-*/=<>^]', x))
    )
    df['constraint_log'] = df['combined_text'].apply(extract_constraint_log)

    keywords = [
        'graph', 'dp', 'tree', 'segment',
        'dijkstra', 'greedy', 'bitwise',
        'modulo', 'combinatorics'
    ]

    for k in keywords:
        df[f'kw_{k}'] = df['cleaned_text'].str.contains(k).astype(int)

    return df

def load_and_preprocess(path):
    data = [json.loads(line) for line in open(path, encoding='utf-8')]
    df = pd.DataFrame(data).fillna("")

    df['combined_text'] = (
        df['title'] + " " +
        df['description'] + " " +
        df['input_description'] * 3
    )

    df['cleaned_text'] = df['combined_text'].apply(clean_problem_text)
    df = extract_features(df)

    tfidf = TfidfVectorizer(
        max_features=1200,
        ngram_range=(1, 2),
        min_df=5
    )
    tfidf_mat = tfidf.fit_transform(df['cleaned_text'])
    joblib.dump(tfidf, 'models/vectorizer.pkl')

    feature_cols = [
        c for c in df.columns if c.startswith('kw_')
    ] + ['text_len', 'math_ops', 'constraint_log']

    scaler = StandardScaler()
    extra = scaler.fit_transform(df[feature_cols])
    joblib.dump(scaler, 'models/feature_scaler.pkl')

    X = hstack([tfidf_mat, extra])

    # 🔥 BINARY TARGET (accuracy boost)
    y = (df['problem_class'].str.lower() == 'hard').astype(int)

    return X, y

if __name__ == "__main__":
    X, y = load_and_preprocess("data/problems_data.jsonl")
    joblib.dump((X, y), "data/processed_data.pkl")
    print("✅ Preprocessing complete (accuracy mode)")
