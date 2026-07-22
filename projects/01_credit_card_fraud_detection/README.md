# 💳 Credit Card Fraud Detection

An end-to-end Machine Learning project for detecting fraudulent credit card transactions using multiple classification algorithms.

The project demonstrates the complete Machine Learning workflow including data preprocessing, handling imbalanced datasets, model training, hyperparameter tuning, threshold optimization, explainability using SHAP, and project modularization into reusable Python modules.

---

# 📌 Problem Statement

Credit card fraud is a highly imbalanced classification problem where fraudulent transactions represent only a very small fraction of all transactions.

The objective of this project is to build and compare multiple machine learning models capable of accurately identifying fraudulent transactions while minimizing false positives.

---

# 📂 Dataset

**Dataset:** Credit Card Fraud Detection Dataset (Kaggle)

Characteristics:

- Transactions made by European cardholders
- 284,807 transactions
- 492 fraud cases
- Fraud ratio ≈ 0.17%
- Highly imbalanced dataset

Features:

- Time
- Amount
- V1 – V28 (PCA transformed features)
- Class (Target)

Target Variable:

- 0 → Genuine Transaction
- 1 → Fraudulent Transaction

---

# 🚀 Project Workflow

```
Raw Dataset
      │
      ▼
Exploratory Data Analysis
      │
      ▼
Train-Test Split
      │
      ▼
Feature Scaling
      │
      ▼
SMOTE (Training Data Only)
      │
      ▼
Model Training
      │
      ▼
Hyperparameter Tuning
      │
      ▼
Model Evaluation
      │
      ▼
Threshold Optimization
      │
      ▼
SHAP Explainability
      │
      ▼
Save Model & Scaler
```

---

# 🛠 Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-Learn
- XGBoost
- SHAP
- Imbalanced-Learn (SMOTE)
- Joblib
- Jupyter Notebook

---

# 📁 Project Structure

```
01_credit_card_fraud_detection/
│
├── artifacts/
│   ├── xgboost.pkl
│   ├── scaler.pkl
│   └── model_metadata.json
│
├── data/
│   ├── raw/
│   └── processed/
│
├── notebooks/
│   ├── EDA.ipynb
│   ├── Model_Training.ipynb
│   └── project_run.ipynb
│
├── src/
│   ├── data/
│   │   ├── load_data.py
│   │   └── preprocess.py
│   │
│   ├── models/
│   │   ├── train_logistic.py
│   │   ├── train_random_forest.py
│   │   ├── train_xgboost.py
│   │   ├── predict.py
│   │   └── model_utils.py
│   │
│   └── evaluation/
│       ├── metrics.py
│       ├── plots.py
│       ├── shap_analysis.py
│       ├── comparison.py
│       └── threshold.py
│
├── requirements.txt
│
└── README.md
```

---

# ⚙️ Data Preprocessing

The following preprocessing steps were performed:

- Missing value inspection
- Duplicate checking
- Train-Test Split
- Feature Scaling using StandardScaler
- Handling class imbalance using SMOTE (training data only)

---

# 🤖 Models Trained

The following models were implemented and compared:

- Logistic Regression
- Logistic Regression (Class Weight Balanced)
- Logistic Regression (SMOTE)
- Random Forest
- Tuned Random Forest
- XGBoost

---

# 📊 Model Evaluation

Models were evaluated using:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC Score
- Confusion Matrix
- Classification Report

---

# 🎯 Threshold Optimization

Instead of relying on the default threshold (0.5), multiple probability thresholds were evaluated.

The project includes reusable utilities for:

- Predicting using custom thresholds
- Threshold evaluation
- Automatic best threshold search

This allows optimization based on business objectives such as maximizing Precision, Recall, or F1 Score.

---

# 📈 Explainable AI (SHAP)

SHAP was used to understand model predictions.

Implemented visualizations:

- SHAP Summary Plot
- SHAP Feature Importance
- SHAP Waterfall Plot

This improves model interpretability by explaining how each feature contributes to fraud predictions.

---

# 📦 Modular Project Design

Instead of keeping all logic inside notebooks, reusable modules were created for:

### Data

- Loading datasets
- Data preprocessing
- Train/Test split
- Scaling
- SMOTE

### Models

- Logistic Regression
- Random Forest
- XGBoost
- Model persistence
- Prediction utilities

### Evaluation

- Metrics
- Plots
- SHAP
- Model comparison
- Threshold optimization

This modular structure improves code readability, maintainability, and reusability.

---

# 💾 Saving Artifacts

The following artifacts are saved after training:

- Trained Model
- Feature Scaler
- Model Metadata

These artifacts can later be used for deployment or inference.

---

# ▶️ Running the Project

Clone the repository

```bash
git clone <repository-url>
```

Create environment

```bash
conda create -n fraud python=3.11
```

Activate

```bash
conda activate fraud
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the notebook

```
notebooks/project_run.ipynb
```

---

# 📌 Future Improvements

Possible extensions include:

- FastAPI deployment
- Docker containerization
- MLflow experiment tracking
- Feature store integration
- Automated retraining pipeline
- Cloud deployment (AWS/GCP/Azure)

---

# 📚 Key Learning Outcomes

Through this project I learned:

- Working with highly imbalanced datasets
- SMOTE oversampling
- Logistic Regression
- Random Forest
- XGBoost
- Hyperparameter tuning
- Threshold optimization
- Model explainability using SHAP
- Modularizing ML projects
- Saving and loading trained models
- Writing reusable ML utilities

---

# 👤 Author

**Aniket Mali**

- IIT Roorkee (Applied Mathematics)
- Data Scientist | Machine Learning Enthusiast

Feel free to connect or provide suggestions for improving the project.