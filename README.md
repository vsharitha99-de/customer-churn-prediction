### **Customer Churn Prediction - Classification**

**Built an end-to-end ML pipeline on `Customer_Transactions.csv` to predict churn and flag at-risk customers for proactive retention.**

#### **Business Problem**
Customer acquisition costs 5x more than retention. This classification model flags customers likely to churn so retention teams can intervene early with targeted offers.

#### **Dataset: Customer_Transactions.csv**
**Target Variable**: `churned` — `0` = Not Churned, `1` = Churned  
**Test Set**: `3000` samples → `2670` non-churn, `330` churn → 11% churn rate

**Key Features Used**:
- **Demographics**: `age`, `gender`, `country` — LabelEncoded
- **Financial**: `annual_income`, `spending_score`, `avg_purchase_value`
- **Behavioral**: `num_purchases`, `membership_years`
- **Dropped**: `customer_id`, `feedback_text`, `last_purchase_date`
- **Data Leakage Fix**: Engineered then dropped `days_since_last_purchase`

#### **Exploratory Data Analysis & Preprocessing**
- Removed duplicates, handled categoricals with `LabelEncoder`
- Identified severe class imbalance in target `churned`
- Applied `SMOTE(random_state=42)` to training data only to balance classes
- `StandardScaler` applied for `KNN` and `SVC`. Tree-based models used unscaled SMOTE data
- Tested feature selection by dropping `age`, `gender`, `country`, `annual_income`, `membership_years`

#### **Model Building & Performance**
Applied 7 classification algorithms with `RandomizedSearchCV` tuning, `cv=5`, `scoring='roc_auc'`. Trained on SMOTE-balanced data, tested on original imbalanced `y_test`:

| Model | Accuracy | Precision (Churn) | Recall (Churn) | F1-Score (Churn) |
| :--- | :---: | :---: | :---: | :---: |
| **KNeighborsClassifier** | 0.70 | 0.21 | 0.63 | 0.32 |
| **SVC** | 0.79 | 0.23 | 0.38 | 0.29 |
| **GaussianNB** | 0.63 | 0.20 | **0.80** | 0.32 |
| **DecisionTreeClassifier** | 0.79 | 0.23 | 0.40 | 0.29 |
| **RandomForestClassifier** | 0.84 | **0.32** | 0.40 | 0.35 |
| **AdaBoostClassifier** | 0.80 | 0.30 | 0.63 | **0.41** |
| **XGBoostClassifier** | **0.86** | 0.30 | 0.23 | 0.26 |

**Best Model for Churn Detection: AdaBoostClassifier**
- **F1-Score = 0.41** and **Recall = 63%** for churn class
- Catches 208 out of 330 actual churners
- **Recall prioritized** because false negatives are costly — missing a churner means lost revenue

**Best Model for Overall Accuracy: XGBoostClassifier**
- **86% Accuracy** but only **23% Recall** on churn class
- Proves accuracy alone is misleading for imbalanced data

#### **Key Insights from Customer_Transactions Data**
1. **SMOTE applied** to training set to improve minority class detection. Test set kept imbalanced for real-world evaluation
2. **Top predictors of churn**: Models relied on `spending_score`, `num_purchases`, `avg_purchase_value`, `membership_years` after feature selection
3. **Business impact**: `GaussianNB` flags 80% of churners, `AdaBoost` gives best balance with 63% recall and 30% precision
4. **Class imbalance effect**: All models have high precision/recall for non-churn `0.91-0.96` but struggle with churn precision `0.20-0.32`

#### **Tech Stack**
`Python`, `Pandas`, `NumPy`, `Scikit-learn`, `XGBoost`, `Imbalanced-learn (SMOTE)`, `Matplotlib`, `Seaborn`
