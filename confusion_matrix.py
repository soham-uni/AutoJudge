import joblib
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Load test data and model
X, y_class, y_score = joblib.load('data/processed_data.pkl')
class_mapping = {'easy':0, 'medium':1, 'hard':2}
y = y_class.str.lower().map(class_mapping)

clf = joblib.load('models/classifier.pkl')

# Split same way as training
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Predict
y_pred = clf.predict(X_test)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

# Display
disp = ConfusionMatrixDisplay(cm, display_labels=["Easy","Medium","Hard"])
disp.plot(cmap="Blues")
plt.title("Confusion Matrix - AutoJudge Classifier")
plt.savefig("confusion_matrix.png", dpi=200)
plt.show()
