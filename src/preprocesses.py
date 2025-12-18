import pandas as pd
import json
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

def clean_problem_text(text):
    text = text.lower()
    
    # 2. Regex Cleaning
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
    
    # 3. Tokenization & Stopword Removal
    stop_words = set(stopwords.words('english'))
    words = text.split()
    
    # 4. Lemmatization: 
    lemmatizer = WordNetLemmatizer()
    cleaned_words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]
    
    return " ".join(cleaned_words)

def load_and_combine(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    
    df = pd.DataFrame(data)
    df = df.fillna("")
    df['combined_text'] = (
        df['title'] + " " + 
        df['description'] + " " + 
        df['input_description'] + " " + 
        df['output_description']
    )

    print("Starting text cleaning (Phase 2)...")
    df['cleaned_text'] = df['combined_text'].apply(clean_problem_text) 
    
    final_df = df[['cleaned_text', 'problem_class', 'problem_score']]
    return final_df

dataset_path = 'data/problems_data.jsonl'
processed_df = load_and_combine(dataset_path)

print("Phase 1 & 2 Complete!")
print("\n--- Raw vs Cleaned Example ---")
print("Original:", processed_df.index[0]) 
print("Cleaned:", processed_df['cleaned_text'].iloc[0][:150], "...") 
print("\nTotal rows:", len(processed_df))