# AutoJudge: Predicting Programming Problem Difficulty ⚖️

**AutoJudge** is an intelligent system built to automatically classify and score programming problems (like those found on Codeforces, Kattis, or CodeChef) based strictly on their textual descriptions. By analyzing the linguistic complexity and domain-specific keywords, the system provides both a categorical difficulty class and a numerical score.

## 🎯 Project Objectives
The goal of this project was to build an automated pipeline that replaces human-dependent judgment with a machine learning model:
* **Predict Difficulty Class**: Classify problems as **Easy, Medium, or Hard**.
* **Predict Difficulty Score**: Assign a precise **numerical difficulty value**.
* **Text-Based Logic**: Work using only textual information from the problem, input, and output descriptions.
* **Web Interface**: Provide a user-friendly UI for real-time predictions.

## 🛠️ Tech Stack
* **Core Logic**: Python 3.x
* **ML Frameworks**: Scikit-learn (Random Forest), XGBoost (Gradient Boosting) 
* **Natural Language Processing**: NLTK (Stopword removal, Lemmatization)
* **Frontend**: Streamlit (Simple Web UI) 
* **Data Handling**: Pandas, NumPy, Scipy (Sparse matrices)

## 🧠 Methodology & Feature Engineering
A "Two-Stage" machine learning pipeline was implemented to handle both classification and regression tasks:

### 1. Data Preprocessing
* **Integration**: Combined Title, Description, Input Description, and Output Description into a single corpus.
* **Cleaning**: Handled missing values and removed linguistic noise using NLTK.

### 2. Feature Extraction
* **TF-IDF Vectors**: Converted text into numerical features using **TF-IDF with Bi-grams**.
* **Custom CP Features**:
    * **Keyword Frequency**: Detection of algorithms like *Graph, DP, Tree, Dijkstra, and Recursion*.
    * **Mathematical Symbol Density**: Counted symbols (+, -, *, ^, etc.) as a proxy for technical complexity.
    * **Text Metadata**: Calculated raw text length to measure description detail.



### 3. Model Architecture
* **Classification**: Utilized a **Random Forest Classifier** to predict the `problem_class`.
* **Regression**: Utilized **XGBoost (Gradient Boosting)** to predict the `problem_score` with higher precision.



## 📊 Evaluation Metrics
The system was evaluated using standard industry metrics:
* **Classification**: Accuracy and Confusion Matrix.
* **Regression**: Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE).

## 💻 Web UI Usage
The interface allows users to:
1. **Paste**: Problem Description, Input Description, and Output Description.
2. **Predict**: Click the "Predict" button to run the real-time inference engine.
3. **View**: Displayed results for the predicted difficulty class and score.



## 📁 Repository Structure
```text
AutoJudge/
├── data/               # Raw dataset and processed .pkl files
├── models/             # Saved trained models (classifier, regressor, vectorizer)
├── src/
│   ├── preprocesses.py # Data cleaning and feature engineering
│   ├── train.py        # Model training and evaluation
│   └── app.py          # Streamlit Web UI
├── requirements.txt    # Project dependencies
└── README.md           # Documentation
