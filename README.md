# 📝 Task 3 – Credit Card Fraud Detection

The goal of this task is to build a machine learning model to detect fraudulent transactions from a highly imbalanced dataset. The dataset contains both legitimate and fraudulent credit card transactions, and the model aims to identify fraud with high precision and recall.

---

## 🚀 Features

### 🧹 Data Preprocessing
- Loaded and explored the credit card transaction dataset
- Handled missing values (if any)
- Scaled numerical features for uniformity
- Maintained dataset structure while preparing for model training

### 🤖 Model Training
- Implemented a classification algorithm (Logistic Regression / Random Forest / XGBoost)
- Addressed class imbalance considerations during training

### 📊 Model Evaluation
- Evaluated models using:
  - Accuracy
  - Precision
  - Recall
  - F1-Score
  - ROC-AUC Score

### 💾 Model Saving
- Generated a Classification Report for detailed performance

---

## 🛠 Tech Stack
- **Python**
- **Pandas**
- **NumPy**
- **scikit-learn**
- **Matplotlib** / **Seaborn**

---

## 📂 Dataset
The dataset contains SMS messages labeled as either `spam` or `ham`.  

**Dataset Source:** [Provided for CodSoft Internship (SMS Spam Collection Dataset)
](https://www.kaggle.com/datasets/kartik2112/fraud-detection)
---

## 📊 Model Workflow
1. Load dataset
2. Preprocess data (cleaning, scaling)
3. Split dataset into training and testing sets
4. Train classification model
5. Evaluate using performance metrics
6. Analyze classification report
---

## 📌 Output Example
Accuracy: 0.9969
Precision: 0.9738
Recall: 0.2079
F1 Score: 0.3427
ROC-AUC: 0.9597

Classification Report:
| Class | Precision | Recall  | F1-Score | Support |
|-------|-----------|---------|----------|---------|
| 0     | 1.00      | 1.00    | 1.00     | 553574  |
| 1     | 0.97      | 0.21    | 0.34     | 2145    |

Macro Avg: Precision = 0.99 | Recall = 0.60 | F1 = 0.67  
Weighted Avg: Precision = 1.00 | Recall = 1.00 | F1 = 1.00
