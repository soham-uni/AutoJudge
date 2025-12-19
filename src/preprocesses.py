import pandas as pd
import json
import re
import nltk
import joblib
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack

# NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

def clean_problem_text(text):
    text = text.lower()
    text = re.sub(r'<.*?>', '', text)
    # Preserve math-critical characters like ^ and *
    text = re.sub(r'[^a-zA-Z0-9\s\^\*]', ' ', text)
    stop_words = set(stopwords.words('english'))
    words = text.split()
    lemmatizer = WordNetLemmatizer()
    return " ".join([lemmatizer.lemmatize(w) for w in words if w not in stop_words])

def extract_custom_features(df):
    # 1. Basic Metadata
    df['text_len'] = df['cleaned_text'].apply(len)
    df['math_symbols'] = df['combined_text'].apply(lambda x: len(re.findall(r'[+\-*/%=<>!^]', x)))
    
    # 2. Advanced Constraint Extraction
    def get_complexity_signals(text):
        # Look for 10^5, 10**9, 2e5, 2 * 10^5
        matches = re.findall(r'(?:10(?:\^|\*\*|e)|1000)\s*(\d+)', text)
        magnitudes = [int(m) for m in matches if m.isdigit()]
        return max(magnitudes) if magnitudes else 0
    
    df['max_constraint'] = df['combined_text'].apply(get_complexity_signals)

    # 3. CP Keywords
    keywords = ['graph', 'dp', 'tree', 'segment', 'dijkstra', 'shortest', 'query', 
                'array', 'string', 'recursion', 'complexity', 'optimal', 'greedy',
                'bitwise', 'modulo', 'combinatorics', 'probability', 'geometry']
    for word in keywords:
        df[f'has_{word}'] = df['cleaned_text'].apply(lambda x: 1 if word in x else 0)
        
    return df

def load_and_preprocess(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    df = pd.DataFrame(data).fillna("")
    
    # WEIGHTED TEXT: Input Description is repeated 3 times to boost constraint importance
    df['combined_text'] = (df['title'] + " ") + df['description'] + (" " + df['input_description']) * 3
    
    df['cleaned_text'] = df['combined_text'].apply(clean_problem_text)
    df = extract_custom_features(df)
    
    # Using Trigams (1, 3) to catch phrases like "least number of"
    tfidf = TfidfVectorizer(max_features=2000, ngram_range=(1, 3)) 
    tfidf_matrix = tfidf.fit_transform(df['cleaned_text'])
    
    joblib.dump(tfidf, 'models/vectorizer.pkl')
    
    # Ensure this list matches the list in app.py exactly
    keyword_list = ['graph', 'dp', 'tree', 'segment', 'dijkstra', 'shortest', 'query', 
                    'array', 'string', 'recursion', 'complexity', 'optimal', 'greedy',
                    'bitwise', 'modulo', 'combinatorics', 'probability', 'geometry']
    
    custom_cols = ['text_len', 'math_symbols', 'max_constraint'] + [f'has_{k}' for k in keyword_list]
    X = hstack([tfidf_matrix, df[custom_cols].values])
    
    return X, df['problem_class'], df['problem_score']

if __name__ == "__main__":
    X, y_class, y_score = load_and_preprocess('data/problems_data.jsonl')
    joblib.dump((X, y_class, y_score), 'data/processed_data.pkl')
    print("Enhanced Weighted Phase 3 Complete!")