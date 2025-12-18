import pandas as pd
import json
import re
import nltk
import joblib  # For saving the TF-IDF vectorizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack

# Ensure NLTK resources are available
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

def clean_problem_text(text):
    text = text.lower()
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
    stop_words = set(stopwords.words('english'))
    words = text.split()
    lemmatizer = WordNetLemmatizer()
    cleaned_words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]
    return " ".join(cleaned_words)

def extract_custom_features(df):
    """
    Phase 3: Extracting metadata and keyword features [cite: 51]
    """
    # 1. Text Length 
    df['text_len'] = df['cleaned_text'].apply(len)
    
    # 2. Keyword Frequency (Binary Flags) 
    keywords = ['graph', 'dp', 'tree', 'segment', 'dijkstra', 'shortest', 'query', 'array', 'string', 'recursion']
    for word in keywords:
        df[f'has_{word}'] = df['cleaned_text'].apply(lambda x: 1 if word in x else 0)
        
    # 3. Math Symbol Count 
    # We check the original text for symbols like +, -, *, /
    df['math_symbols'] = df['combined_text'].apply(lambda x: len(re.findall(r'[+\-*/%=<>!^]', x)))
    
    return df

def load_and_preprocess(file_path):
    # Phase 1: Load and Combine 
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    df = pd.DataFrame(data).fillna("")
    
    df['combined_text'] = df['title'] + " " + df['description'] + " " + df['input_description'] + " " + df['output_description'] 
    
    # Phase 2: Clean 
    print("Cleaning text...")
    df['cleaned_text'] = df['combined_text'].apply(clean_problem_text)
    
    # Phase 3: Custom Features 
    print("Extracting custom features...")
    df = extract_custom_features(df)
    
    # Phase 3: TF-IDF Vectorization 
    print("Generating TF-IDF vectors...")
    tfidf = TfidfVectorizer(max_features=1000) # Keep top 1000 words
    tfidf_matrix = tfidf.fit_transform(df['cleaned_text'])
    
    # Save vectorizer for the Web UI later
    joblib.dump(tfidf, 'models/vectorizer.pkl')
    
    # Combine TF-IDF with custom features
    custom_feats = df[['text_len', 'math_symbols'] + [f'has_{k}' for k in ['graph', 'dp', 'tree', 'segment', 'dijkstra', 'shortest', 'query', 'array', 'string', 'recursion']]].values
    X = hstack([tfidf_matrix, custom_feats])
    
    return X, df['problem_class'], df['problem_score']

if __name__ == "__main__":
    dataset_path = 'data/problems_data.jsonl'
    X, y_class, y_score = load_and_preprocess(dataset_path)
    
    # Save the processed features and labels for Phase 4
    joblib.dump((X, y_class, y_score), 'data/processed_data.pkl')
    print("Phase 3 Complete! Processed data saved in data/processed_data.pkl")