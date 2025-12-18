import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, mean_absolute_error
from xgboost import XGBRegressor # You may need to pip install xgboost
from sklearn.metrics import classification_report, mean_squared_error
import numpy as np

# 1. Load the processed features and targets from Phase 3
X, y_class, y_score = joblib.load('data/processed_data.pkl')

# 2. Convert class labels (Easy/Medium/Hard) to numbers for the model
# Easy = 0, Medium = 1, Hard = 2
class_mapping = {'easy': 0, 'medium': 1, 'hard': 2}
y_class_num = y_class.str.lower().map(class_mapping)

# 3. Split data for Classification
X_train, X_test, y_c_train, y_c_test = train_test_split(X, y_class_num, test_size=0.2, random_state=42)

# 4. Train Model 1: Classification (Random Forest)
print("Training Classification Model...")
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_c_train)

# 5. Train Model 2: Regression (XGBoost)
# We split separately for regression targets
X_train_r, X_test_r, y_s_train, y_s_test = train_test_split(X, y_score, test_size=0.2, random_state=42)

print("Training Regression Model (XGBoost)...")
reg = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5)
reg.fit(X_train_r, y_s_train)

# 6. Save the models
joblib.dump(clf, 'models/classifier.pkl')
joblib.dump(reg, 'models/regressor.pkl')

print("Phase 4 Complete! Models saved in models/ folder.")

# 7. Evaluation - Classification
y_c_pred = clf.predict(X_test)
print("\n--- Classification Report ---")
print(classification_report(y_c_test, y_c_pred, target_names=['Easy', 'Medium', 'Hard']))

# 8. Evaluation - Regression
y_s_pred = reg.predict(X_test_r)
mae = mean_absolute_error(y_s_test, y_s_pred)
rmse = np.sqrt(mean_squared_error(y_s_test, y_s_pred))

print(f"--- Regression Metrics ---")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")