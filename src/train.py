import joblib
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import classification_report, mean_absolute_error, mean_squared_error
from xgboost import XGBClassifier, XGBRegressor
from imblearn.over_sampling import SMOTE

# Load data
X, y_class, y_score = joblib.load('data/processed_data.pkl')

class_mapping = {'easy': 0, 'medium': 1, 'hard': 2}
y_class_num = y_class.str.lower().map(class_mapping)

# Balance the dataset
smote = SMOTE(random_state=42)
X_bal, y_bal = smote.fit_resample(X, y_class_num)

# Split
X_train, X_test, y_c_train, y_c_test = train_test_split(
    X_bal, y_bal, test_size=0.2, random_state=42, stratify=y_bal
)

X_train_r, X_test_r, y_s_train, y_s_test = train_test_split(
    X, y_score, test_size=0.2, random_state=42
)

# ================= CLASSIFIER =================
print("\n--- Training XGBoost Classifier ---")

clf = XGBClassifier(
    n_estimators=400,
    learning_rate=0.05,
    max_depth=8,
    subsample=0.9,
    colsample_bytree=0.9,
    eval_metric='mlogloss',
    random_state=42
)

clf.fit(X_train, y_c_train)

# ================= REGRESSOR =================
print("\n--- Training XGBoost Regressor ---")

param_dist = {
    'n_estimators': [400, 600, 800],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [6, 8, 10],
    'subsample': [0.7, 0.85, 1.0],
    'colsample_bytree': [0.7, 0.85, 1.0],
}

random_search = RandomizedSearchCV(
    XGBRegressor(random_state=42),
    param_distributions=param_dist,
    n_iter=12,
    cv=3,
    scoring='neg_mean_absolute_error',
    n_jobs=-1,
    verbose=1
)

random_search.fit(X_train_r, y_s_train)
reg = random_search.best_estimator_

# Save models
joblib.dump(clf, 'models/classifier.pkl')
joblib.dump(reg, 'models/regressor.pkl')

# ================= EVALUATION =================

print("\n================ FINAL RESULTS ================")

y_pred = clf.predict(X_test)
print("\nClassification Report:")
print(classification_report(y_c_test, y_pred, target_names=['Easy', 'Medium', 'Hard']))

y_s_pred = reg.predict(X_test_r)
mae = mean_absolute_error(y_s_test, y_s_pred)
rmse = np.sqrt(mean_squared_error(y_s_test, y_s_pred))

print("\nRegression:")
print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
