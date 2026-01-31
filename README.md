# Customer Purchase Behavior Prediction – Data Mining Project

## 📌 Project Overview

This project applies a complete **Data Mining pipeline** to predict whether a customer will make a purchase based on demographic and behavioral data.  
The study is carried out in a **marketing context**, where predicting customer purchase behavior is a key factor for decision-making and campaign optimization.

The project follows all classical stages of Data Mining: data exploration, preprocessing, modeling, optimization, evaluation, and interpretation of results.

---

## 🎯 Objectives

- Predict customer purchase behavior (binary classification)
- Compare multiple supervised classification algorithms
- Optimize model performance using hyperparameter tuning
- Identify key factors influencing purchasing decisions
- Support marketing decision-making through data-driven insights

---

## 📂 Dataset Description

The dataset is sourced from **Kaggle** and contains demographic and behavioral information about customers interacting with an online platform.

### Main Features
- **Age**: Customer age  
- **Gender**: 0 (Male), 1 (Female)  
- **Annual Income**: Customer annual income  
- **Number of Purchases**: Total number of previous purchases  
- **Product Category**:  
  - 0: Electronics  
  - 1: Clothing  
  - 2: Household  
  - 3: Beauty  
  - 4: Sports  
- **Time Spent on Website**: Time in minutes  
- **Loyalty Program**: 0 (No), 1 (Yes)  
- **Discounts Availed**: Number of discounts used (0–5)  
- **PurchaseStatus (Target)**: 0 (No purchase), 1 (Purchase)

---

## 🔍 Exploratory Data Analysis

Exploratory analysis was conducted to:
- Understand variable distributions
- Detect anomalies and outliers
- Analyze correlations between features
- Identify class imbalance in the target variable

The analysis revealed that customers who spend more time on the website and those enrolled in loyalty programs are more likely to make purchases.

---

## 🧹 Data Preparation

The following preprocessing steps were applied:
- Handling missing and anomalous values
- Normalization of numerical features
- Encoding of categorical variables
- Train-test split to ensure reliable evaluation

These steps improved data quality and model robustness.

---

## 🤖 Classification Models Used

Several supervised learning algorithms were implemented and compared:
- Logistic Regression
- K-Nearest Neighbors (KNN)
- Support Vector Machine (SVM)
- Random Forest
- Gradient Boosting

Each model was initially trained with default parameters to establish baseline performance.

---

## 📊 Experimental Results

Model performance was evaluated using:
- Accuracy
- Precision
- Recall
- F1-score

Tree-based models (Random Forest and Gradient Boosting) achieved the best results, particularly in terms of F1-score, making them more suitable for handling class imbalance.

---

## ⚙️ Hyperparameter Optimization

To improve performance, **GridSearchCV** and **RandomizedSearchCV** were applied to the best-performing models.  
This optimization significantly improved model accuracy while reducing overfitting through cross-validation.

---

## 🏆 Final Model Selection

The **Random Forest** model was selected as the final model based on:
- High F1-score
- Robustness
- Balanced performance and complexity

Final evaluation included:
- Confusion matrix
- Classification report
- ROC curve analysis

---

## 📈 Marketing Insights

Feature importance analysis shows that:
- Number of previous purchases
- Loyalty program membership

have a strong influence on purchase probability.  
These insights can be directly used to improve customer targeting and personalize marketing campaigns.

---

## ⚠️ Project Limitations

- Limited dataset size
- Potential class imbalance
- Missing external behavioral variables

---

## 🔮 Future Improvements

- Integration of additional customer data
- Use of advanced models (XGBoost, Neural Networks)
- Deployment in a real-time marketing system
- Automation of the prediction pipeline

---

## 🛠️ Tools & Technologies

- Python
- Pandas, NumPy
- Scikit-learn
- Matplotlib, Seaborn
- Jupyter Notebook

---

## 📄 License

Academic project for educational purposes.
