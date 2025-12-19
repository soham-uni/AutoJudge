import joblib
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, mean_absolute_error, mean_squared_error  
from xgboost import XGBRegressor

# 1. Load the enhanced features from Phase 3
X, y_class, y_score = joblib.load('data/processed_data.pkl') 

# 2. Map classes to numbers
class_mapping = {'easy': 0, 'medium': 1, 'hard': 2} 
y_class_num = y_class.str.lower().map(class_mapping)

# 3. Data Split (80/20)
X_train, X_test, y_c_train, y_c_test = train_test_split(X, y_class_num, test_size=0.2, random_state=42)
X_train_r, X_test_r, y_s_train, y_s_test = train_test_split(X, y_score, test_size=0.2, random_state=42)

# 4. Train Model 1: Classification (Random Forest)
print("Training Classification Model (Random Forest)...")  
clf = RandomForestClassifier(n_estimators=200, max_depth=20, random_state=42)
clf.fit(X_train, y_c_train)

# 5. Train Model 2: Regression (XGBoost with Hyperparameter Tuning)
print("Starting Hyperparameter Tuning for XGBoost...") 

# Define the grid of parameters to test
param_grid = {
    'n_estimators': [100, 200],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 5, 7],
    'subsample': [0.8, 1.0]
}

xgb_model = XGBRegressor(random_state=42)
grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, 
                           cv=3, scoring='neg_mean_absolute_error', verbose=1)

grid_search.fit(X_train_r, y_s_train)

# Use the best model found by the grid search
reg = grid_search.best_estimator_
print(f"Best Parameters Found: {grid_search.best_params_}")

# 6. Save Optimized Models
joblib.dump(clf, 'models/classifier.pkl')
joblib.dump(reg, 'models/regressor.pkl')

# 7. Evaluation
y_c_pred = clf.predict(X_test)
print("\n--- Classification Report ---")  
print(classification_report(y_c_test, y_c_pred, target_names=['Easy', 'Medium', 'Hard']))

y_s_pred = reg.predict(X_test_r)
mae = mean_absolute_error(y_s_test, y_s_pred) 
rmse = np.sqrt(mean_squared_error(y_s_test, y_s_pred)) 

print(f"\n--- Optimized Regression Metrics ---")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")