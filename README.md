# Loan Prediction Project

## Project Overview
This project focuses on building a machine learning model to predict loan approval status based on various applicant and loan-related features. The goal is to assist financial institutions in making informed decisions regarding loan applications, thereby reducing risk and improving efficiency in the loan approval process.

## Problem Statement
In the financial industry, accurately assessing the creditworthiness of loan applicants is crucial to minimize financial losses due to defaults. Manual assessment can be time-consuming and prone to human error. There is a need for an automated system that can reliably predict whether a loan application will be approved or rejected based on historical data.

## Project Objective
The primary objective of this project is to develop and evaluate a machine learning model capable of predicting loan approval status (`Loan_Status`) with high accuracy. This involves:
* Performing Exploratory Data Analysis (EDA) to understand the dataset characteristics and identify key relationships.
* Handling missing values and outliers in the dataset.
* Transforming categorical features into numerical format suitable for machine learning algorithms.
* Training a classification model (Support Vector Classifier) to predict loan status.
* Evaluating the model's performance using appropriate metrics such as accuracy, precision, recall, and F1-score.

## Column Dictionary

| Column Name        | Description                                                               | Data Type (Original) |
|:-------------------|:--------------------------------------------------------------------------|:---------------------|
| `Loan_ID`          | Unique identifier for each loan application.                              | Object               |
| `Gender`           | Applicant's gender (Male/Female).                                         | Object               |
| `Married`          | Applicant's marital status (Yes/No).                                      | Object               |
| `Dependents`       | Number of dependents the applicant has (0, 1, 2, 3+).                     | Object               |
| `Education`        | Applicant's education level (Graduate/Not Graduate).                      | Object               |
| `Self_Employed`    | Whether the applicant is self-employed (Yes/No).                          | Object               |
| `ApplicantIncome`  | Applicant's monthly income.                                               | Integer              |
| `CoapplicantIncome`| Co-applicant's monthly income.                                            | Float                |
| `LoanAmount`       | Loan amount in thousands.                                                 | Float                |
| `Loan_Amount_Term` | Term of loan in months.                                                   | Float                |
| `Credit_History`   | Credit history meets guidelines (1: Yes, 0: No).                          | Float                |
| `Property_Area`    | Location of the property (Urban/Semiurban/Rural).                         | Object               |
| `Loan_Status`      | Loan approved (Y) or not (N) - **Target Variable**.                       | Object               |

## Methodology and Steps Taken

1.  **Data Loading and Initial Inspection:**
    * The `loan_prediction.csv` dataset was loaded into a Pandas DataFrame.
    * Initial inspection revealed 614 entries and 13 columns.
    * Identified columns with missing values: `Gender`, `Married`, `Dependents`, `Self_Employed`, `LoanAmount`, `Loan_Amount_Term`, and `Credit_History`.

2.  **Data Preprocessing and Cleaning:**
    * **Irrelevant Feature Removal:** The `Loan_ID` column was dropped as it serves merely as an identifier and holds no predictive power.
    * **Handling Missing Values:**
        * Missing values in categorical columns (`Gender`, `Married`, `Dependents`, `Self_Employed`, `Loan_Amount_Term`, `Credit_History`) were imputed using the **mode** of their respective columns.
        * Missing values in the numerical `LoanAmount` column were imputed using the **median** of the column to mitigate the effect of potential outliers.
    * **Outlier Treatment:**
        * Outliers in `ApplicantIncome` were removed using the Interquartile Range (IQR) method (values outside $Q1 - 1.5 \times IQR$ and $Q3 + 1.5 \times IQR$).
        * Outliers in `CoapplicantIncome` were also removed using the IQR method.

3.  **Exploratory Data Analysis (EDA) - Visualizations and Insights:**
    * **Loan Status Distribution:** The target variable `Loan_Status` was found to be imbalanced, with a significantly higher number of approved loans (`Y`) compared to rejected loans (`N`). This highlights the need to consider imbalance handling techniques during model development.
    * **Gender Distribution:** The dataset is predominantly composed of male applicants.
    * **Marital Status Distribution:** A majority of the loan applicants are married.
    * **Education Distribution:** Most applicants are Graduates, indicating a higher proportion of educated individuals in the applicant pool.
    * **Self-Employment Distribution:** The vast majority of applicants are not self-employed.
    * **Applicant Income Distribution:** The original distribution was heavily right-skewed, with many applicants having lower incomes and a few high-income outliers. After outlier removal, the distribution became much less skewed.
    * **Loan Status vs. Applicant Income:** While both approved and rejected loans show wide income ranges, the median applicant income for approved loans appears slightly higher.
    * **Loan Status vs. Coapplicant Income:** A large number of applicants in both approved and rejected categories have zero coapplicant income. For those with coapplicant income, the distribution is still right-skewed, and coapplicant income alone doesn't appear to be a strong differentiating factor.
    * **Loan Status vs. Loan Amount:** The median loan amount for approved loans is slightly higher than for rejected loans. Both distributions are positively skewed.
    * **Loan Status vs. Credit History:** This was identified as a highly influential factor. Applicants with a credit history of 1 (good credit) are overwhelmingly likely to get their loan approved, whereas those with a credit history of 0 (bad credit) are largely rejected.
    * **Loan Status vs. Property Area:** Loans in Semiurban areas appear to have the highest approval rate, followed by Urban, then Rural areas.

4.  **Feature Engineering and Scaling:**
    * **One-Hot Encoding:** Categorical features (`Gender`, `Married`, `Dependents`, `Education`, `Self_Employed`, `Property_Area`) were converted into numerical format using one-hot encoding.
    * **Feature Scaling:** Numerical features (`ApplicantIncome`, `CoapplicantIncome`, `LoanAmount`, `Loan_Amount_Term`, `Credit_History`) were scaled using `StandardScaler` to ensure that no single feature dominates the model training due to its scale.

5.  **Model Training:**
    * The dataset was split into training (80%) and testing (20%) sets.
    * A **Support Vector Classifier (SVC)** model was chosen and trained on the scaled training data.

6.  **Model Evaluation:**
    * Predictions (`y_pred`) were made on the test set.
    * The model's performance was evaluated using:
        * **Accuracy:** 0.8273
        * **Classification Report:**
            * Precision (N): 0.94, Recall (N): 0.49, F1-score (N): 0.64
            * Precision (Y): 0.80, Recall (Y): 0.99, F1-score (Y): 0.89
        * **Confusion Matrix:**
            * True Negatives (Correctly Rejected): 17
            * False Positives (Incorrectly Approved): 2
            * False Negatives (Incorrectly Rejected): 1
            * True Positives (Correctly Approved): 74

## Final Insights

* The dataset is imbalanced towards loan approvals, which the model learned to prioritize, resulting in very high recall for approved loans.
* `Credit_History` is by far the most significant predictor of loan status. A positive credit history strongly correlates with loan approval.
* Applicants from Semiurban areas show a higher propensity for loan approval.
* While income features (`ApplicantIncome`, `CoapplicantIncome`) are important, their distributions, even after outlier handling, show significant overlap between approved and rejected categories, suggesting they are not standalone determinants.
* The model excels at identifying loans that *will be approved* (high recall for 'Y') and at correctly identifying when it predicts a loan *will be rejected* (high precision for 'N').
* However, the model has a notable weakness in identifying *all* loans that should be rejected (low recall for 'N'). This means it incorrectly predicts a significant portion of actual rejections as approvals (false negatives).

## Recommendations

1.  **Address Class Imbalance:** Given the high recall for 'Y' and low recall for 'N', consider employing techniques like oversampling (e.g., SMOTE) or undersampling on the training data to balance the classes. This could improve the model's ability to correctly identify rejected loans.
2.  **Feature Engineering:**
    * Create a `TotalIncome` feature by combining `ApplicantIncome` and `CoapplicantIncome`. This might provide a more holistic view of the applicant's financial capacity.
    * Derive `LoanAmount_per_Income` to understand the loan burden relative to income.
3.  **Explore Other Models:** While SVC performs reasonably, experiment with other classification algorithms such as Logistic Regression, Decision Trees, Random Forests, or Gradient Boosting (e.g., XGBoost, LightGBM). These models might offer different trade-offs in precision and recall, and some are less sensitive to class imbalance or feature scaling.
4.  **Hyperparameter Tuning:** Conduct thorough hyperparameter tuning for the chosen model (e.g., using GridSearchCV or RandomizedSearchCV for SVC) to optimize its performance further.
5.  **Cost-Sensitive Learning:** If the cost of a false positive (approving a bad loan) is higher than a false negative (rejecting a good loan), consider using cost-sensitive learning techniques or adjusting the model's decision threshold.

## Conclusion

This project successfully established a baseline machine learning model for loan prediction. The EDA provided valuable insights into the dataset, particularly highlighting the dominance of `Credit_History` and the class imbalance in loan approvals. The trained SVC model achieved good overall accuracy, demonstrating strong capability in predicting loan approvals. However, there's room for improvement, especially in correctly identifying all loan rejections. By implementing the suggested recommendations, the model's predictive power and robustness can be further enhanced for more reliable loan decision-making.
