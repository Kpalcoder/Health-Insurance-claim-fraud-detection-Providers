# 🏥 Healthcare Insurance Fraud Detection using Machine Learning

## 📘 Project Overview
Healthcare insurance fraud is a growing concern, resulting in massive financial losses every year.  
This project uses **Machine Learning (ML)** to automatically detect fraudulent claims based on patterns in healthcare data such as patient demographics, claim amounts, hospital stay durations, and provider behaviors.

The system is designed using **Python-based scripts** that perform:
- Data exploration (EDA)
- Feature engineering
- Model development and evaluation  
to build a robust and scalable **fraud detection pipeline**.

---

## 🎯 Project Goals
- Detect fraudulent insurance claims using historical data  
- Build an automated ML pipeline from raw data to model prediction  
- Optimize accuracy while minimizing false positives  
- Provide interpretable metrics for fraud investigators  

---

## 📂 Project Structure

## ⚙️ Technologies Used
- **Python 3.x**
- **Libraries:**  
  `pandas`, `numpy`, `scikit-learn`, `xgboost`, `matplotlib`, `seaborn`, `imbalanced-learn`, `joblib`
- **Tools:**  
  Jupyter Notebook, VS Code, GitHub  

---

## 🧠 Project Workflow

### 1️⃣ Data Exploration (`eda_analysis.py`)
- Import and clean raw healthcare claim data  
- Handle missing values and outliers  
- Analyze claim trends, fraud distributions, and correlations  
- Generate visual insights (bar plots, heatmaps, distributions)

### 2️⃣ Feature Engineering (`feature_engineering.py`)
- Encode categorical features using OneHot or Label Encoding  
- Scale numerical data using StandardScaler  
- Create derived variables (e.g., claim ratio, patient age group)  
- Apply **SMOTE** for class imbalance correction  

### 3️⃣ Model Development (`model_development.py`)
- Split data into training and testing sets  
- Train models such as:
  - Logistic Regression  
  - Random Forest  
  - XGBoost  
- Perform hyperparameter tuning (GridSearchCV)  
- Evaluate performance using:
  - Accuracy
  - Precision
  - Recall
  - F1 Score
  - ROC-AUC Curve  
- Save the best-performing model using `joblib`  

### 4️⃣ Prediction (`predict.py`)
- Load the trained model (`trained_model.pkl`)
- Predict fraud probability on new unseen claim data
- Output results with claim ID and fraud prediction label  

---

## 📊 Example Results
| Model              | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|--------------------|-----------|------------|---------|-----------|----------|
| Random Forest      | 0.9029     | 0.48       | 0.83    | 0.61      | 0.95     |
| XGBoost            | 0.9297      | 0.65       | 0.52    | 0.58      | 0.93     |


---

## 🔧 Installation & Setup

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/<Kpalcoder>/healthcare-fraud-detection.git
cd healthcare-fraud-detection

