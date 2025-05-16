# Employee Attrition Prediction Project

This project aims to predict employee attrition based on historical data.
It includes data preprocessing, model training, and a web interface for uploading new data and getting predictions.

## Project Structure

- `data/`: To store datasets. The primary dataset (`Danh sách nhân sự.xlsx`) is expected here for `train_model.py`.
- `notebook/`: To present what methodology overview
- `src/`: Source code for the project.
  - `preprocessing.py`: Script for data loading, cleaning, feature engineering, and preprocessing.
  - `train_model.py`: Script for training models, evaluating them, performing SHAP analysis, and saving the best model and reports.
  - `app.py`: Flask application for the web interface.
  - `templates/`: Contains HTML templates for the web interface (`index.html`).
  - `uploads/`: Temporary storage for files uploaded via the web app.
  - `downloads/`: Temporary storage for prediction result files to be downloaded.
- `models/`: To store trained machine learning models, preprocessor artifacts, reports, and SHAP plots.
  - `best_attrition_model.joblib`: The best performing trained model.
  - `preprocessor_artifacts.joblib`: Saved preprocessor objects (scaler, encoders, maps, selected features).
  - `model_comparison_report.txt`: Text file comparing performance of different models.
  - `shap_summary_plot.png`: Image showing SHAP feature importance for the best model.
- `requirements.txt`: Python dependencies for the project.
- `.gitignore`: Specifies intentionally untracked files.
- `README.md`: This file.

## Methodology Overview

1.  **Data Loading:** Loads data from the Excel file.
2.  **Initial Cleaning:** Drops irrelevant or problematic columns (identifiers, data leakage columns, columns with excessive missing values).
3.  **Feature Engineering:**
    *   Calculates `Độ tuổi` (Age) from `Ngày sinh` (Date of Birth).
    *   Extracts Year, Month, Day components from date columns (`Ngày vào làm`, `Ngày kết thúc thử việc`, `Ngày bắt đầu HĐ`).
    *   Calculates `ProbationDurationDays` from `Ngày vào làm` and `Ngày kết thúc thử việc`.
4.  **Target Variable Creation:** The `Đang làm việc` column is transformed into a binary target variable (1 for attrited, 0 for active).
5.  **Encoding & Imputation:**
    *   **Target Encoding:** Applied to key categorical features like `Phòng Ban`, `Chức danh`, `Địa Điểm Làm Việc`, `Công Ty`. The mean of the target variable for each category (calculated on the training set) is used. Unseen categories during prediction are handled using the global target mean.
    *   **Label Encoding:** Applied to remaining categorical features.
    *   **Imputation:** Missing numerical values are imputed with the median, and categorical values with the mode (or "Unknown").
6.  **Scaling:** Numerical features (including target/frequency encoded ones) are scaled using `StandardScaler`.
7.  **Feature Selection (SHAP-based):** After initial SHAP analysis on a model trained with all engineered features, the top 20 most impactful features were selected to train the final models. This helps in reducing noise and potentially improving generalization.
8.  **Model Training & Evaluation:**
    *   Several classification algorithms are trained (e.g., Logistic Regression, Random Forest, Gradient Boosting).
    *   The Random Forest model (with `class_weight='balanced'`) currently yields the best performance.
    *   Models are evaluated on a hold-out test set using accuracy, classification report (precision, recall, F1-score), and confusion matrix.
9.  **SHAP Analysis:** SHAP (SHapley Additive exPlanations) values are calculated for the best model to understand feature contributions to predictions. A summary plot is saved.
10. **Artifact Saving:** The best model, preprocessor objects (including scalers, encoders, target mean maps, selected feature list), and evaluation reports are saved.

## Dataset Insights

* The primary dataset used is "Danh sách nhân sự.xlsx".
* The target variable is derived from the "Đang làm việc" column (1 for currently working, blank for has left).
* The initial input data consists of **8,702 employee records** with **43 types of information**.
* After processing and feature selection, the model is trained using **20 types of information (features)** for these **8,702 employees**.

### Feature Importances (SHAP)

--- Mean Absolute SHAP Values (Top 10) ---

| Feature               | Mean Absolute SHAP Value |
|-----------------------|--------------------------|
| Chức danh             | 0.108584                 |
| Ngày bắt đầu HĐ_Month | 0.049523                 |
| Số tháng review gần   | 0.049311                 |
| Địa Điểm Làm Việc     | 0.045597                 |
| Tổng Mức Lương        | 0.034394                 |
| Ngày bắt đầu HĐ_Year  | 0.029116                 |
| Thâm niên             | 0.023076                 |
| Mức thưởng KPI hoặc Khoán | 0.022094                 |
| Mức Phụ cấp           | 0.017186                 |
| Loại hợp đồng         | 0.016623                 |

# Model Comparison Report

## **Best Model**: Random Forest

-   Accuracy: 0.8966
-   Confusion Matrix:

    |   | T   | F   |
    |---|-----|-----|
    | T | 550 | 54  |
    | F | 126 | 1011|

**Classification Report:**

|           | precision | recall | f1-score | support |
|-----------|-----------|--------|----------|---------|
| 0         | 0.81      | 0.91   | 0.86     | 604     |
| 1         | 0.95      | 0.89   | 0.92     | 1137    |
| accuracy  |           |        | 0.90     | 1741    |
| macro avg | 0.88      | 0.90   | 0.89     | 1741    |
| weighted avg| 0.90      | 0.90   | 0.90     | 1741    |


## Model: Logistic Regression

-   Accuracy: 0.8208
-   Confusion Matrix:

    |   | T   | F   |
    |---|-----|-----|
    | T | 504 | 100 |
    | F | 212 | 925 |

**Classification Report:**

|           | precision | recall | f1-score | support |
|-----------|-----------|--------|----------|---------|
| 0         | 0.70      | 0.83   | 0.76     | 604     |
| 1         | 0.90      | 0.81   | 0.86     | 1137    |
| accuracy  |           |        | 0.82     | 1741    |
| macro avg | 0.80      | 0.82   | 0.81     | 1741    |
| weighted avg| 0.83      | 0.82   | 0.82     | 1741    |


## Model: Gradient Boosting

-   Accuracy: 0.8817
-   Confusion Matrix:

    |   | T   | F    |
    |---|-----|------|
    | T | 519 | 85   |
    | F | 121 | 1016 |

**Classification Report:**

|           | precision | recall | f1-score | support |
|-----------|-----------|--------|----------|---------|
| 0         | 0.81      | 0.86   | 0.83     | 604     |
| 1         | 0.92      | 0.89   | 0.91     | 1137    |
| accuracy  |           |        | 0.88     | 1741    |
| macro avg | 0.87      | 0.88   | 0.87     | 1741    |
| weighted avg| 0.88      | 0.88   | 0.88     | 1741    |

# Web Interface
An HTML interface (`index.html`) is provided via a Flask application (`app.py`):
1.  Allows uploading an Excel file.
    *   If the file contains a "Đang làm việc" column, predictions are made only for rows where this is 1 (active).
    *   If the column is absent, predictions are made for all rows.
2.  The application uses the saved best model and preprocessor to make predictions.
3.  Displays a summary of predictions.
4.  Allows downloading an Excel file containing all original uploaded data plus new columns for `AttritionPrediction` and `AttritionProbability (%)`.

## Setup and Usage

1.  **Clone the repository (if applicable).**
2.  **Create a Python virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **Place your dataset:** Ensure your primary dataset (e.g., `Danh sách nhân sự.xlsx`) is in the `employee_attrition_predictor/data/` directory. Update `DATA_FILE_PATH` in `src/train_model.py` and `src/inspect_data.py` if your filename or location differs.
5.  **Train the model (if not already done or if data changes):**
    Navigate to the `src` directory and run:
    ```bash
    cd employee_attrition_predictor/src
    python train_model.py
    ```
    This will save `best_attrition_model.joblib`, `preprocessor_artifacts.joblib`, `model_comparison_report.txt`, and `shap_summary_plot.png` in the `models/` directory.
6.  **Run the web application:**
    From the `src` directory:
    ```bash
    python app.py
    ```
    Access the application in your browser, typically at `http://127.0.0.1:5000/`.
    To allow access from other machines on your local network, run: `python app.py --host=0.0.0.0` (ensure your firewall allows connections to port 5000).

