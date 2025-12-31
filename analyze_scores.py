import pandas as pd
import joblib

# Load processed data
X, y_class, y_score = joblib.load('data/processed_data.pkl')

# Build DataFrame
df = pd.DataFrame({
    "Class": y_class.str.lower(),
    "Score": y_score
})

# Print detailed statistics
print("\nDifficulty Score Distribution by Class:\n")
print(df.groupby("Class")["Score"].describe())
