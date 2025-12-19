import joblib
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBRegressor
from sklearn.metrics import classification_report, mean_absolute_error, mean_squared_error

# 1. Load data
X, y_class, y_score = joblib.load('data/processed_data.pkl')
class_mapping = {'easy': 0, 'medium': 1, 'hard': 2}
y_class_num = y_class.str.lower().map(class_mapping)

# Stratify ensures the model sees a balanced mix of difficulties during training
X_train, X_test, y_c_train, y_c_test = train_test_split(
    X, y_class_num, test_size=0.2, random_state=42, stratify=y_class_num
)
X_train_r, X_test_r, y_s_train, y_s_test = train_test_split(
    X, y_score, test_size=0.2, random_state=42
)

# 2. Optimized Classifier (Random Forest)
print("--- Step 1: Training Classifier ---")
clf = RandomForestClassifier(
    n_estimators=300, 
    max_depth=25, 
    class_weight='balanced', 
    n_jobs=-1, # Fast training
    random_state=42
)
clf.fit(X_train, y_c_train)

# 3. Deep Tuning Regressor (XGBoost)
print("--- Step 2: Deep Tuning XGBoost (This might take a few minutes) ---")

# We explore a wider range of values for more accuracy
param_dist = {
    'n_estimators': [300, 500, 700],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'max_depth': [4, 6, 8, 10],
    'subsample': [0.7, 0.8, 0.9],
    'colsample_bytree': [0.7, 0.8, 0.9],
    'gamma': [0, 0.1, 0.2] # Complexity control
}

random_search = RandomizedSearchCV(
    XGBRegressor(tree_method='hist', random_state=42), 
    param_distributions=param_dist,
    n_iter=15, # Testing 15 diverse combinations
    cv=3, 
    scoring='neg_mean_absolute_error',
    n_jobs=-1, # Use all CPU cores to help your laptop finish faster
    verbose=1
)

random_search.fit(X_train_r, y_s_train)
reg = random_search.best_estimator_

# 4. Save the "Brains"
joblib.dump(clf, 'models/classifier.pkl')
joblib.dump(reg, 'models/regressor.pkl')

# 5. Final Evaluation
print("\n" + "="*30)
print("FINAL MODEL PERFORMANCE")
print("="*30)

y_c_pred = clf.predict(X_test)
print("\nClassification Report:")
print(classification_report(y_c_test, y_c_pred, target_names=['Easy', 'Medium', 'Hard']))

y_s_pred = reg.predict(X_test_r)
mae = mean_absolute_error(y_s_test, y_s_pred)
rmse = np.sqrt(mean_squared_error(y_s_test, y_s_pred))

print(f"\nRegression Metrics:")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"Best Params Found: {random_search.best_params_}")