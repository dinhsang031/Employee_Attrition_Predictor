import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
# Removed XGBClassifier import
# Removed SMOTE import
import numpy as np # Added for potential use with coefficients
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import os
import shap # Import SHAP
import matplotlib.pyplot as plt # For saving SHAP plots
from preprocessing import load_data, preprocess_data # Updated import

# Define paths
DATA_FILE_PATH = r"C:\Users\Dell03.LAPTOP-PTNNLMOI\Desktop\employee_attrition_predictor\data\Danh sách nhân sự.xlsx"
MODEL_DIR = '../models/' # Relative to src directory
BEST_MODEL_PATH = os.path.join(MODEL_DIR, 'best_attrition_model.joblib')
# PREPROCESSOR_PATH is now managed by preprocess_data, but we define it for clarity where it's saved
PREPROCESSOR_ARTIFACTS_PATH = os.path.join(MODEL_DIR, 'preprocessor_artifacts.joblib')


def train_and_evaluate_models(X_train, y_train, X_test, y_test, feature_names): # Added feature_names
    """
    Trains multiple models, evaluates them, and returns the best one.
    """
    models = {
        'Logistic Regression': LogisticRegression(solver='liblinear', random_state=42, class_weight='balanced'),
        'Random Forest': RandomForestClassifier(random_state=42), # Added class_weight back
        'Gradient Boosting': GradientBoostingClassifier(random_state=42),
        # 'SVC': SVC(random_state=42, probability=True) # Keep SVC commented for now
    }

    best_model = None
    best_accuracy = 0
    model_results = {}

    print("\n--- Model Training and Evaluation (Default Parameters) ---")
    for name, model_instance in models.items():
        print(f"\nTraining {name}...")
        model_instance.fit(X_train, y_train)

        y_pred_test = model_instance.predict(X_test)
        accuracy_test = accuracy_score(y_test, y_pred_test)
        report_str_test = classification_report(y_test, y_pred_test)
        
        print(f"Test Set Accuracy for {name}: {accuracy_test:.4f}")
        print("Test Set Classification Report:")
        print(report_str_test)
        print("Test Set Confusion Matrix:")
        print(confusion_matrix(y_test, y_pred_test))

        model_results[name] = {
            'model_estimator': model_instance,
            'accuracy_test': accuracy_test,
            'classification_report_str_test': report_str_test,
            'best_params': "Default" 
        }

        if accuracy_test > best_accuracy:
            best_accuracy = accuracy_test
            best_model = model_instance
            print(f"New best model (on test set): {name} with accuracy: {best_accuracy:.4f}")

    if best_model:
        print(f"\nOverall Best Performing Model (on test set): {type(best_model).__name__} with accuracy: {best_accuracy:.4f}")
    else:
        print("No models were trained successfully.")

    # Save comparison report
    report_content = "=== Model Comparison Report (Default Parameters) ===\n\n"
    for name, result in model_results.items():
        report_content += f"--- Model: {name} ---\n"
        # report_content += f"Best Parameters: {result['best_params']}\n" # Not showing 'Default' for cleaner report
        report_content += f"Test Set Accuracy: {result['accuracy_test']:.4f}\n"
        report_content += "Test Set Classification Report:\n"
        report_content += result['classification_report_str_test'] + "\n\n"
        
    report_path = os.path.join(MODEL_DIR, 'model_comparison_report.txt')
    try:
        if not os.path.exists(MODEL_DIR):
            os.makedirs(MODEL_DIR)
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        print(f"Model comparison report saved to {report_path}")
    except Exception as e:
        print(f"Error saving model comparison report: {e}")

    return best_model, model_results

def find_important_features(model, X_data_for_shap, feature_names): # Added X_data_for_shap
    """
    Calculates and prints SHAP values for feature importance.
    Saves a SHAP summary plot.
    """
    print("\n--- Calculating SHAP Values for Feature Importance ---")
    explainer = None
    shap_values = None

    # SHAP can be slow on very large datasets for some explainers.
    # Consider using a sample of X_data_for_shap if performance is an issue.
    # For TreeExplainer, it's usually efficient.
    # X_data_sample = shap.sample(X_data_for_shap, 100) if len(X_data_for_shap) > 1000 else X_data_for_shap
    X_data_sample = X_data_for_shap # Using full X_test for now

    try:
        if isinstance(model, (RandomForestClassifier, GradientBoostingClassifier)): # XGBClassifier removed from this line
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_data_sample)
        elif isinstance(model, LogisticRegression):
            # For LinearExplainer, it expects the model and the data used for training the masker (can be X_train)
            # Or, more simply, pass the coefficient and intercept if available and model is simple.
            # Using KernelExplainer for a more general approach if LinearExplainer is tricky with preprocessor.
            # However, TreeExplainer is preferred for tree models.
            # For Logistic Regression, coefficients are often clear enough.
            # Let's try LinearExplainer with X_data_sample as background.
            explainer = shap.LinearExplainer(model, X_data_sample)
            shap_values = explainer.shap_values(X_data_sample)
        else:
            print(f"SHAP explainer not specifically configured for {type(model).__name__}. Using KernelExplainer (might be slow).")
            # KernelExplainer needs a summary of the background data (e.g., k-means)
            # For simplicity, if not tree or linear, we might fall back or skip.
            # For now, let's just print a message if it's not a covered model type.
            print(f"Skipping SHAP for model type: {type(model).__name__} as it's not explicitly handled (e.g. SVC).")
            return None

        if shap_values is not None:
            print(f"Type of shap_values: {type(shap_values)}")
            if isinstance(shap_values, list):
                print(f"Length of shap_values list: {len(shap_values)}")
                for i, sv_item in enumerate(shap_values):
                    print(f"Shape of shap_values[{i}]: {sv_item.shape if hasattr(sv_item, 'shape') else 'N/A'}")
            elif hasattr(shap_values, 'shape'):
                print(f"Shape of shap_values: {shap_values.shape}")

            # For binary classification, shap_values can be a list of two arrays (one for each class)
            # or a single array if the explainer handles it.
            # We are interested in the impact on the positive class (attrition = 1)
            shap_values_for_report = shap_values # Default assignment

            if isinstance(shap_values, list) and len(shap_values) == 2: # For explainers returning list for binary classification
                shap_values_for_report = shap_values[1] 
            elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3 and shap_values.shape[2] == 2: # For explainers returning 3D array (samples, features, classes)
                shap_values_for_report = shap_values[:, :, 1] # Select SHAP values for class 1
            # If shap_values is already 2D (e.g. for LinearExplainer if model outputs single value), it's used as is.
            
            print(f"Type of shap_values_for_report (after selection): {type(shap_values_for_report)}")
            if hasattr(shap_values_for_report, 'shape'):
                print(f"Shape of shap_values_for_report (after selection): {shap_values_for_report.shape}")

            # Ensure shap_values_for_report is a 2D numpy array here (samples, features)
            if not (isinstance(shap_values_for_report, np.ndarray) and shap_values_for_report.ndim == 2):
                print(f"Error: shap_values_for_report is not a 2D numpy array as expected. Actual ndim: {shap_values_for_report.ndim if hasattr(shap_values_for_report, 'ndim') else 'N/A'}")
                return None # Cannot proceed if not 2D
            
            mean_abs_shap = np.abs(shap_values_for_report).mean(axis=0) # Should now be 1D
            
            print(f"Shape of mean_abs_shap (should be 1D): {mean_abs_shap.shape if hasattr(mean_abs_shap, 'shape') else 'N/A'}")
            print(f"Length of feature_names: {len(feature_names)}")
            
            if mean_abs_shap.ndim != 1 or len(mean_abs_shap) != len(feature_names):
                print(f"Error: Mismatch between mean_abs_shap dimensions ({mean_abs_shap.ndim}, len {len(mean_abs_shap) if hasattr(mean_abs_shap, 'shape') else 'N/A'}) and feature_names (len {len(feature_names)}).")
                return None


            shap_importance_df = pd.DataFrame({
                'feature': feature_names, # Should be 1D list or array
                'mean_abs_shap_value': mean_abs_shap # Should be 1D array of same length
            })
            shap_importance_df = shap_importance_df.sort_values(by='mean_abs_shap_value', ascending=False)
            
            print("\n--- Mean Absolute SHAP Values (Top 10) ---")
            print(shap_importance_df.head(10))

            # Generate and save SHAP summary plot (bar plot)
            plt.figure()
            shap.summary_plot(shap_values_for_report, X_data_sample, plot_type="bar", feature_names=feature_names, show=False)
            plot_path = os.path.join(MODEL_DIR, 'shap_summary_plot.png')
            plt.savefig(plot_path, bbox_inches='tight')
            plt.close() # Close the plot to free memory
            print(f"SHAP summary bar plot saved to {plot_path}")
            
            return shap_importance_df

    except Exception as e:
        print(f"Error calculating or plotting SHAP values: {e}")
        return None


def save_model(model, model_path):
    """Saves the trained model."""
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
    if model:
        joblib.dump(model, model_path)
        print(f"\nModel saved to {model_path}")

if __name__ == '__main__':
    print("Starting model training process...")
    raw_df = load_data(DATA_FILE_PATH)

    if raw_df is not None:
        print("\nRaw data loaded successfully.")
        
        # Use the updated preprocess_data function
        # It now handles saving its own preprocessor artifacts (scaler, encoders, etc.)
        X_processed, y_processed, feature_names_out = preprocess_data(
            raw_df.copy(), 
            is_training=True, 
            preprocessor_objects_path=PREPROCESSOR_ARTIFACTS_PATH
        )

        if X_processed is not None and y_processed is not None:
            if X_processed.empty:
                print("Error: Feature set X_processed is empty after preprocessing. Check preprocessing steps and data.")
            else:
                # Ensure all data in X_processed is numeric before splitting and training
                # This should ideally be guaranteed by preprocess_data, but as a safeguard:
                non_numeric_cols = X_processed.select_dtypes(exclude=np.number).columns
                if not non_numeric_cols.empty:
                    print(f"Warning: Non-numeric columns found after preprocessing: {non_numeric_cols.tolist()}")
                    print("Attempting to convert them to numeric. This indicates an issue in preprocess_data.")
                    for col in non_numeric_cols:
                        X_processed[col] = pd.to_numeric(X_processed[col], errors='coerce').fillna(0) # Coerce and fill
                
                # Check for NaNs again after all processing, though imputation should handle it
                if X_processed.isnull().sum().sum() > 0:
                    print("Warning: NaNs found in X_processed before training. Filling with 0. Check imputation in preprocess_data.")
                    X_processed = X_processed.fillna(0)


                X_train, X_test, y_train, y_test = train_test_split(
                    X_processed, y_processed, test_size=0.2, random_state=42, stratify=y_processed
                )
                # SMOTE block removed
                print(f"\nData split into training and testing sets.") # Adjusted print message
                print(f"X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")
                print(f"y_train shape: {y_train.shape}, y_test shape: {y_test.shape}")


                # Convert X_test to DataFrame if it's not, for SHAP (it should be from preprocess_data)
                if not isinstance(X_test, pd.DataFrame):
                    X_test_df = pd.DataFrame(X_test, columns=feature_names_out)
                else:
                    X_test_df = X_test

                # Train models on original (but split) training data
                best_model, model_results_dict = train_and_evaluate_models(X_train, y_train, X_test_df, y_test, feature_names_out)

                if best_model:
                    # Pass X_test_df (or a sample) to find_important_features for SHAP
                    find_important_features(best_model, X_test_df, feature_names_out)
                    save_model(best_model, BEST_MODEL_PATH) # Only save the model here
                else:
                    print("\nModel training failed or no best model was identified.")
        else:
            print("\nData preprocessing failed. Cannot proceed with model training.")
            print("Please check the `preprocess_data` function in `preprocessing.py` and the dataset.")
    else:
        print("\nData loading failed. Cannot proceed.")

    print("\nModel training script finished.")
