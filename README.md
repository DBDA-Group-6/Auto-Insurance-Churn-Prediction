# Auto Insurance Churn Prediction System

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![SQL](https://img.shields.io/badge/SQL-MySQL-4479A1.svg)
![ML](https://img.shields.io/badge/ML-XGBoost%20%7C%20CatBoost-orange.svg)

> **Production-ready Flask + ML pipeline for predicting customer churn in auto insurance using AWS EC2 & RDS**

End-to-end system that ingests CSV files, preprocesses data (including fixed OHE for city/county), applies SMOTE + undersampling, predicts churn probability, stores results in MySQL, and provides Tableau visualizations. Achieves strong Recall & F1-Score on imbalanced data.

### [View Tableau Dashboard](https://public.tableau.com/views/Auto_insurance_17665565943760/NumberofCustomersChurnedBymartialStatusandHasChildren/1?:language=en-US&:display_count=n&:origin=viz_share_link)

---

## Project Overview

**Business Problem:** Identify customers likely to churn (cancel their auto insurance policy) so the company can take proactive retention actions and reduce revenue loss.

**Solution:** Built a complete web + ML system with:
- Flask web app for CSV upload & prediction
- Consistent preprocessing pipeline (Pandas)
- SMOTE + undersampling for imbalance
- Multiple classifiers + ensemble boosters
- AWS EC2 deployment + RDS MySQL storage
- Tableau dashboard for churn insights

**Impact:**
- Strong focus on **Recall** (~85%) and **F1-Score** (~82%) to catch most churners
- Fixed OHE issue for consistent inference on new data
- Deployed on free-tier AWS EC2 t3.small with RDS
- Interactive results view, download, and Tableau analytics

---

## Key Results

| Model                  | Recall   | F1-Score | Accuracy | Log Loss | Notes                              |
|------------------------|----------|----------|----------|----------|------------------------------------|
| **XGBoost**            | **~85%** | **~82%** | —        | Low      | **Selected model** – best balance  |
| CatBoost               | ~84%     | ~81%     | —        | Low      | Excellent categorical handling     |
| Logistic Regression    | —        | —        | Baseline | —        | Simple & interpretable baseline    |
| Bernoulli Naive Bayes  | —        | —        | —        | —        | Suited to binary OHE features      |

**Selected Model:** XGBoost (highest F1 & Recall, robust to imbalance, fast inference)

## Technical Pipeline

### Data Management
- **Input**: Raw CSV file uploaded via web interface
- **Storage**: AWS RDS MySQL (`prediction` table)
- **Target**: `Churn` (binary: **1** = churn, **0** = retain)

### Preprocessing (Key Steps)
1. Drop irrelevant columns (IDs, dates, lat/long, state)
2. Label encode `home_market_value` (ordinal mapping)
3. Cap outliers using IQR × 1.5 on numeric columns
4. Fill missing values (median for numeric, mode for categorical)
5. Fixed one-hot encoding for `city` (~100 values), `county`, `marital_status`
6. SMOTE oversampling + majority undersampling → final ratio **4:8** (minority:majority)
7. Ensure consistent column order between training and inference

### Model Training
- **Class imbalance handling**: SMOTE + undersampling (4:8 ratio for stability)
- **Primary evaluation metrics**: Recall & F1-Score  
  (Dataset is imbalanced with ~20–30% churn rate)
- **Validation strategy**: Train-test split + cross-validation
- **Model saving format**: `.joblib`

### Deployment
- **Platform**: AWS EC2 t3.small (free tier)
- **Database**: AWS RDS MySQL
- **Web framework**: Flask
- **Core features**:
  - CSV file upload
  - Real-time prediction
  - View & download prediction results
  - Health check endpoint
- **Version control**: Git + GitHub (with SSH setup script for EC2)

## Model Selection Justification

### Why XGBoost? (Selected Model)
- Highest balance of **F1-Score** and **Recall**
- Robust to class imbalance and noisy/outlier-prone features
- Fast training and inference — ideal for free-tier EC2
- Excellent performance with post-OHE binary features
- Hyperparameters carefully tuned for stability  
  (subsample, colsample_bytree, learning_rate, max_depth, etc.)

### Comparison Insights
- **CatBoost**  
  Very close performance, strong categorical handling, but slightly slower in our setup
- **Logistic Regression**  
  Fast, highly interpretable baseline — but lower predictive power
- **Bernoulli Naive Bayes**  
  Extremely quick, well-suited to binary OHE columns — but ignores feature interactions

## Technology Stack

**Backend & Machine Learning**  
- Python (Flask, Pandas, scikit-learn, xgboost, catboost, joblib)  
- MySQL (via SQLAlchemy + pymysql)

**Deployment**  
- AWS EC2 (t3.small – free tier)  
- AWS RDS (MySQL)

**Frontend**  
- HTML + CSS (responsive upload & results pages)

**Visualization**  
- Tableau Public (embedded interactive dashboards)

**Version Control**  
- Git + GitHub (with automated SSH setup script)

## Key Learnings
- Fixed one-hot encoding columns are **essential** for consistent inference on new data
- In churn problems, **Recall** and **F1-Score** are far more important than accuracy
- Tuning SMOTE + undersampling ratio (4:8) significantly improved model stability
- Free-tier EC2 storage limits forced switch from PySpark → Pandas
- Tree-based boosters (especially **XGBoost**) clearly outperform simple models on this dataset
