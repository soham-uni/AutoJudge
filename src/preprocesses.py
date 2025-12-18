import pandas as pd
import json

def load_and_combine(file_path):
    # 1. Load the JSONL file
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    
    df = pd.DataFrame(data)
    
    # 2. Handle missing values (if any) [cite: 32]
    df = df.fillna("")
    
    # 3. Combine text fields into one single text input [cite: 52]
    # We combine title, description, input_description, and output_description [cite: 16, 17, 18, 19]
    df['combined_text'] = (
        df['title'] + " " + 
        df['description'] + " " + 
        df['input_description'] + " " + 
        df['output_description']
    )
    
    # 4. Filter for only what we need for the model
    # Features: combined_text
    # Targets: problem_class (classification) and problem_score (regression) [cite: 4, 5]
    final_df = df[['combined_text', 'problem_class', 'problem_score']]
    
    return final_df

# Usage
dataset_path = 'data/problems_data.jsonl'
processed_df = load_and_combine(dataset_path)

print("Phase 1 Complete!")
print(processed_df.head())
print(len(processed_df), "rows processed.")