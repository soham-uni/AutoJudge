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
    # Keep numbers and powers like 10^5
    text = re.sub(r'[^a-zA-Z0-9\s\^]', ' ', text)
    stop_words = set(stopwords.words('english'))
    words = text.split()
    lemmatizer = WordNetLemmatizer()
    cleaned_words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]
    return " ".join(cleaned_words)

def extract_custom_features(df):
    """
    Enhanced Feature Engineering [cite: 51, 53]
    """
    # 1. Text Length [cite: 54]
    df['text_len'] = df['cleaned_text'].apply(len)
    
    # 2. Keyword Frequency (Updated List) 
    keywords = ['graph', 'dp', 'tree', 'segment', 'dijkstra', 'shortest', 'query', 
                'array', 'string', 'recursion', 'complexity', 'optimal', 'greedy']
    for word in keywords:
        df[f'has_{word}'] = df['cleaned_text'].apply(lambda x: 1 if word in x else 0)
        
    # 3. Math Symbol Count [cite: 55]
    df['math_symbols'] = df['combined_text'].apply(lambda x: len(re.findall(r'[+\-*/%=<>!^]', x)))
    
    # 4. Constraint Extraction (Improving accuracy by finding specific magnitudes)
    # Checks for common CP constraints like 10^5, 10^9, 100000, etc.
    def get_max_constraint(text):
        matches = re.findall(r'10\^(\d+)|1000+', text)
        if not matches:
            return 0
        # Convert scientific notation or count zeros to find magnitude
        magnitudes = [int(m) if m.isdigit() else len(m) for m in matches if m]
        return max(magnitudes) if magnitudes else 0

    df['max_constraint_magnitude'] = df['combined_text'].apply(get_max_constraint)
    
    return df

def load_and_preprocess(file_path):
    # Phase 1: Load and Combine [cite: 31, 52]
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    df = pd.DataFrame(data).fillna("")
    
    df['combined_text'] = (
        df['title'] + " " + 
        df['description'] + " " + 
        df['input_description'] + " " + 
        df['output_description']
    ) 
    
    # Phase 2: Clean
    print("Cleaning text...")
    df['cleaned_text'] = df['combined_text'].apply(clean_problem_text)
    
    # Phase 3: Custom Features
    print("Extracting enhanced custom features...")
    df = extract_custom_features(df)
    
    # Phase 3: TF-IDF Vectorization (Updated to Bigrams) 
    print("Generating TF-IDF Bigram vectors...")
    # Increase max_features to 1500 and include bigrams (ngram_range=(1, 2))
    tfidf = TfidfVectorizer(max_features=1500, ngram_range=(1, 2)) 
    tfidf_matrix = tfidf.fit_transform(df['cleaned_text'])
    
    # Save vectorizer for the Web UI
    joblib.dump(tfidf, 'models/vectorizer.pkl')
    
    # Combine TF-IDF with custom features
    keyword_cols = [f'has_{k}' for k in ['graph', 'dp', 'tree', 'segment', 'dijkstra', 
                                         'shortest', 'query', 'array', 'string', 
                                         'recursion', 'complexity', 'optimal', 'greedy']]
    
    meta_cols = ['text_len', 'math_symbols', 'max_constraint_magnitude']
    
    custom_feats = df[meta_cols + keyword_cols].values
    X = hstack([tfidf_matrix, custom_feats])
    
    return X, df['problem_class'], df['problem_score']

if __name__ == "__main__":
    dataset_path = 'data/problems_data.jsonl'
    X, y_class, y_score = load_and_preprocess(dataset_path)
    
    # Save the processed features and labels for Phase 4
    joblib.dump((X, y_class, y_score), 'data/processed_data.pkl')
    print("Phase 3 Complete! Enhanced data saved in data/processed_data.pkl")