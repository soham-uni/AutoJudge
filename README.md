# AutoJudge — Programming Problem Difficulty Predictor

## Project Overview

AutoJudge is an intelligent machine learning system that automatically predicts the difficulty of competitive programming problems using only their textual description.

It predicts:
- **Problem Class:** Easy / Medium / Hard *(Classification task)*
- **Problem Score:** A continuous numerical difficulty value in the range **0–10** *(Regression task)*

The goal is to replicate how online judges (Codeforces, CodeChef, Kattis, etc.) estimate problem difficulty — but automatically, without human feedback.

---

## 📂 Dataset Used

The dataset is derived from the public benchmark:

**TaskComplexityEval-24**  
https://github.com/AREEG94FAHAD/TaskComplexityEval-24

Each problem sample contains:
- title  
- description  
- input_description  
- output_description  
- problem_class (Easy / Medium / Hard)  
- problem_score (numerical difficulty)

After preprocessing, the cleaned dataset is stored locally as:

data/processed_data.pkl

---

## 🧠 Approach & Models Used

### Data Preprocessing
- Combined all text fields into a single input
- Text cleaning, normalization, stopword removal, lemmatization
- Handled missing values

### Feature Extraction
Each problem is converted into a numerical feature vector using:
- TF-IDF Vectorization (1–3 n-grams)
- Text length
- Mathematical symbol frequency
- Constraint detection (e.g., 10^5, 10^9, etc.)
- Competitive programming keyword indicators (graph, dp, tree, greedy, dijkstra, etc.)

### Models

Task | Model  
Classification | XGBoost Classifier  
Regression | XGBoost Regressor  
Text Representation | TF-IDF Vectorizer  

To handle class imbalance, SMOTE was applied only to the classifier training set.  
The regression model was trained on the original dataset.

---

## 📊 Evaluation Metrics

Classification:
- Accuracy: ~67%

Regression:
- MAE: 1.67  
- RMSE: 2.00  

The classifier and regressor are trained independently.  
---

## 🖥️ Web Interface

A clean and modern Streamlit interface allows users to:
1. Paste the full problem statement  
2. Click Analyze Complexity  
3. Instantly view:
   - Predicted difficulty class  
   - Predicted difficulty score  

The interface runs locally and is designed for live demonstration.

---

## ⚙️ Steps to Run the Project Locally
> Pre-trained models are already included in this repository.

### 1️⃣ Install Dependencies
```bash
pip install -r requirements/requirements
```

### 2️⃣ Launch Web Interface
```bash
streamlit run src/app.py
```

Open in your browser:
```
http://localhost:8501
```

### 🧪 Optional: Retraining the Models

```bash
python src/preprocesses.py
python src/train.py
```

---
## 🎥 Demo Video

Demo Link: https://youtu.be/PA4KqOCAP2M

The video demonstrates:
- Project overview  
- Model approach  
- Working web interface with predictions  

---

## 📁 Repository Structure

```text
AUTOJUDGE/
├── data/
│   ├── problems_data.jsonl
│   └── processed_data.pkl
│
├── models/
│   ├── classifier.pkl
│   ├── regressor.pkl
│   └── vectorizer.pkl
│
├── requirements/
│   └── requirements
├── results/
│
├── src/
│   ├── app.py
│   ├── preprocesses.py
│   └── train.py
│
├── analyze_scores.py
├── .gitignore
└── README.md
```

---

## 👤 Author

Soham Adak  
IIT Roorkee
