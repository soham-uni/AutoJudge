import joblib
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report

X, y = joblib.load("data/processed_data.pkl")

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

param_grid = {
    'n_estimators': [400, 600, 800],
    'max_depth': [5, 7, 9],
    'learning_rate': [0.03, 0.05, 0.1],
    'subsample': [0.7, 0.85],
    'colsample_bytree': [0.7, 0.85]
}

search = RandomizedSearchCV(
    XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        tree_method='hist',
        scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum(),
        random_state=42
    ),
    param_grid,
    n_iter=20,
    cv=3,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)

search.fit(X_train, y_train)
clf = search.best_estimator_

joblib.dump(clf, "models/classifier.pkl")

y_pred = clf.predict(X_test)

print("\n=== CLASSIFICATION REPORT (Hard vs Not-Hard) ===")
print(classification_report(y_test, y_pred, target_names=['Easy/Medium', 'Hard']))
print(f"\n🔥 Accuracy: {accuracy_score(y_test, y_pred):.3f}")
