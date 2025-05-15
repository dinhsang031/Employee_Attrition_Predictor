import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
# Removed PCA import
import joblib # To save preprocessor objects

def load_data(file_path):
    """Loads data from an Excel file."""
    try:
        df = pd.read_excel(file_path, engine='openpyxl')
        print(f"Successfully loaded data from {file_path}")
        return df
    except FileNotFoundError:
        print(f"Error: The file {file_path} was not found.")
        return None
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def preprocess_data(df, is_training=True, preprocessor_objects_path=None):
    """
    Preprocesses the raw data for training or prediction.

    Args:
        df (pd.DataFrame): The input dataframe.
        is_training (bool): True if preprocessing for training, False for prediction.
        preprocessor_objects_path (str, optional): Path to load/save preprocessor objects.
                                                   Required if is_training=False.

    Returns:
        If is_training=True:
            X (pd.DataFrame): Processed features.
            y (pd.Series): Processed target variable.
            feature_names_out (list): List of feature names after preprocessing.
        If is_training=False:
            X_processed (pd.DataFrame): Processed features for prediction.
            feature_names_out (list): List of feature names after preprocessing.
    """
    if df is None:
        return None if is_training else (None, None)

    print(f"\n--- Starting Preprocessing (is_training={is_training}) ---")
    df_processed = df.copy()

    # 1. Handle Target Variable (only during training)
    if is_training:
        if 'Đang làm việc' not in df_processed.columns:
            print("Error: Target column 'Đang làm việc' not found in training data.")
            return None, None, None
        # 1 means working (target=0), NaN/blank means left (target=1)
        df_processed['target'] = df_processed['Đang làm việc'].apply(lambda x: 0 if x == 1 or str(x).lower() == '1' else 1)
        print(f"Target variable 'target' created. Distribution:\n{df_processed['target'].value_counts(normalize=True)}")
    
    # 2. Define columns to drop
    cols_to_drop = [
        'Đang làm việc',  # Original target, or not present in prediction data
        'Mã Nhân Viên', 'Họ Và Tên', 'Số điện thoại', 'Email',  # Identifiers
        'Ngày nghỉ việc', 'Lý do nghỉ việc',  # Data leakage / Post-attrition
        'Trùng Kiêm nhiệm',  # Mostly missing
        'Khối', # Mostly missing
        'Trình độ ngoại ngữ', 'Tin học', # Mostly missing
        'Ngày kết thúc HĐ', # Mostly missing
        'Ngày thâm niên', # Redundant if 'Thâm niên' is used
        # 'Ngày sinh' will be dropped after 'Độ tuổi' is processed
    ]
    # Drop columns if they exist
    existing_cols_to_drop = [col for col in cols_to_drop if col in df_processed.columns]
    df_processed = df_processed.drop(columns=existing_cols_to_drop)
    print(f"Dropped columns: {existing_cols_to_drop}")

    # 3. Feature Engineering & Type Conversion
    # Convert 'Ngày sinh' to datetime and calculate 'Độ tuổi'
    if 'Ngày sinh' in df_processed.columns:
        # Handle various date formats if necessary, attempt common ones
        df_processed['Ngày sinh'] = pd.to_datetime(df_processed['Ngày sinh'], errors='coerce')
        current_year = pd.Timestamp('now').year
        df_processed['Độ tuổi'] = current_year - df_processed['Ngày sinh'].dt.year
        df_processed = df_processed.drop(columns=['Ngày sinh'])
        print("'Độ tuổi' calculated from 'Ngày sinh'. 'Ngày sinh' dropped.")
    elif 'Độ tuổi' in df_processed.columns: # If 'Độ tuổi' already exists
        df_processed['Độ tuổi'] = pd.to_numeric(df_processed['Độ tuổi'], errors='coerce')
        print("'Độ tuổi' converted to numeric.")

    # Date feature engineering: Extract year, month, day from relevant date columns
    # Also, calculate probation duration
    original_date_cols = {} # To store original datetime columns for duration calculation
    
    if 'Ngày vào làm' in df_processed.columns:
        df_processed['Ngày vào làm'] = pd.to_datetime(df_processed['Ngày vào làm'], errors='coerce')
        original_date_cols['Ngày vào làm'] = df_processed['Ngày vào làm'].copy()
        df_processed['Ngày vào làm_Year'] = df_processed['Ngày vào làm'].dt.year
        df_processed['Ngày vào làm_Month'] = df_processed['Ngày vào làm'].dt.month
        df_processed['Ngày vào làm_Day'] = df_processed['Ngày vào làm'].dt.day
        # df_processed = df_processed.drop(columns=['Ngày vào làm']) # Keep for duration, drop later
        print(f"Extracted Year, Month, Day from 'Ngày vào làm'.")

    if 'Ngày kết thúc thử việc' in df_processed.columns:
        df_processed['Ngày kết thúc thử việc'] = pd.to_datetime(df_processed['Ngày kết thúc thử việc'], errors='coerce')
        original_date_cols['Ngày kết thúc thử việc'] = df_processed['Ngày kết thúc thử việc'].copy()
        df_processed['Ngày kết thúc thử việc_Year'] = df_processed['Ngày kết thúc thử việc'].dt.year
        df_processed['Ngày kết thúc thử việc_Month'] = df_processed['Ngày kết thúc thử việc'].dt.month
        df_processed['Ngày kết thúc thử việc_Day'] = df_processed['Ngày kết thúc thử việc'].dt.day
        # df_processed = df_processed.drop(columns=['Ngày kết thúc thử việc']) # Keep for duration, drop later
        print(f"Extracted Year, Month, Day from 'Ngày kết thúc thử việc'.")

    if 'Ngày bắt đầu HĐ' in df_processed.columns: # Contract Start Date
        df_processed['Ngày bắt đầu HĐ'] = pd.to_datetime(df_processed['Ngày bắt đầu HĐ'], errors='coerce')
        # original_date_cols['Ngày bắt đầu HĐ'] = df_processed['Ngày bắt đầu HĐ'].copy() # Not used for duration yet
        df_processed['Ngày bắt đầu HĐ_Year'] = df_processed['Ngày bắt đầu HĐ'].dt.year
        df_processed['Ngày bắt đầu HĐ_Month'] = df_processed['Ngày bắt đầu HĐ'].dt.month
        df_processed['Ngày bắt đầu HĐ_Day'] = df_processed['Ngày bắt đầu HĐ'].dt.day
        df_processed = df_processed.drop(columns=['Ngày bắt đầu HĐ'])
        print(f"Extracted Year, Month, Day from 'Ngày bắt đầu HĐ' and dropped original.")

    # Calculate ProbationDurationDays
    if 'Ngày vào làm' in original_date_cols and 'Ngày kết thúc thử việc' in original_date_cols:
        df_processed['ProbationDurationDays'] = (original_date_cols['Ngày kết thúc thử việc'] - original_date_cols['Ngày vào làm']).dt.days
        print("'ProbationDurationDays' calculated.")
    else:
        df_processed['ProbationDurationDays'] = np.nan # Fill with NaN if dates are missing
        print("'ProbationDurationDays' set to NaN as one or both required dates are missing.")

    # Now drop the original full date columns if they were kept for duration calculation
    if 'Ngày vào làm' in df_processed.columns and 'Ngày vào làm' in original_date_cols : df_processed = df_processed.drop(columns=['Ngày vào làm'])
    if 'Ngày kết thúc thử việc' in df_processed.columns and 'Ngày kết thúc thử việc' in original_date_cols : df_processed = df_processed.drop(columns=['Ngày kết thúc thử việc'])


    # 4. Define X and y (if training)
    if is_training:
        X = df_processed.drop(columns=['target'], errors='ignore')
        y = df_processed['target']
    else:
        X = df_processed

    # Columns for Target Encoding
    cols_for_target_encoding = ['Phòng Ban', 'Chức danh', 'Địa Điểm Làm Việc', 'Công Ty']
    # Columns for Frequency Encoding (if any, ensure no overlap with target encoding)
    cols_for_freq_encoding = [] # Example: ['SomeOtherColumn'] - currently none as we prioritize Target Encoding for main ones
    
    # Load or Initialize Preprocessor Objects
    if is_training:
        label_encoders = {} 
        frequency_maps = {}
        target_mean_maps = {} # For Target Encoding
        global_target_mean = 0 # For Target Encoding
        scaler = StandardScaler()
        imputation_values = {'numerical': {}, 'categorical': {}}
    else:
        if not preprocessor_objects_path or not joblib.os.path.exists(preprocessor_objects_path):
            print(f"Error: Preprocessor objects path '{preprocessor_objects_path}' not found for prediction.")
            return None, None
        try:
            saved_objects = joblib.load(preprocessor_objects_path)
            label_encoders = saved_objects.get('label_encoders', {})
            frequency_maps = saved_objects.get('frequency_maps', {})
            target_mean_maps = saved_objects.get('target_mean_maps', {}) # Load Target Encoding maps
            global_target_mean = saved_objects.get('global_target_mean', 0.5) # Load global mean, default 0.5
            scaler = saved_objects['scaler']
            imputation_values = saved_objects['imputation_values']
            expected_feature_names = saved_objects.get('feature_names_out', X.columns.tolist())
            
            X_reordered = pd.DataFrame(columns=expected_feature_names)
            for col in expected_feature_names:
                if col in X.columns:
                    X_reordered[col] = X[col]
                else:
                    X_reordered[col] = np.nan 
            X = X_reordered[expected_feature_names]
            print("Loaded preprocessor objects and reordered columns for prediction.")
        except Exception as e:
            print(f"Error loading preprocessor objects: {e}")
            return None, None

    # Initial identification of categorical and numerical columns (before freq encoding changes types)
    temp_numerical_cols = X.select_dtypes(include=np.number).columns.tolist()
    temp_categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()

    # Impute BEFORE frequency encoding for categorical columns being frequency encoded
    print("\n--- Imputation (Round 1 - Before Freq Encoding) ---")
    for col in temp_numerical_cols:
        if col in X.columns: # Check if column still exists (it should)
            if is_training:
                median_val = X[col].median()
                imputation_values['numerical'][col] = median_val
                X[col] = X[col].fillna(median_val)
            else:
                median_val = imputation_values['numerical'].get(col, 0) # Default to 0 if not found
                X[col] = X[col].fillna(median_val)
    
    for col in temp_categorical_cols:
        if col in X.columns:
            if is_training:
                mode_val = X[col].mode()[0] if not X[col].mode().empty else "Unknown"
                imputation_values['categorical'][col] = mode_val
                X[col] = X[col].fillna(mode_val)
            else:
                mode_val = imputation_values['categorical'].get(col, "Unknown") # Default to "Unknown"
                X[col] = X[col].fillna(mode_val)

    # Target Encoding for specified columns
    # This is done before re-identifying numerical/categorical columns as it converts categorical to numerical
    print("\n--- Target Encoding ---")
    if is_training:
        global_target_mean = y.mean() # Calculate global mean of target for handling unseen categories

    for col in cols_for_target_encoding:
        if col in X.columns:
            if is_training:
                # Calculate mean of target for each category in the column
                # Concatenate X[col] and y to ensure proper alignment for groupby().mean()
                target_mean_map = pd.concat([X[col], y], axis=1).groupby(col)['target'].mean()
                target_mean_maps[col] = target_mean_map
                X[col] = X[col].map(target_mean_map)
                print(f"Categorical column '{col}': TargetEncoded.")
            else: # Prediction
                if col in target_mean_maps:
                    target_mean_map = target_mean_maps[col]
                    X[col] = X[col].map(target_mean_map)
                    # For categories in test not seen in train, map will result in NaN. Fill with global mean.
                    X[col] = X[col].fillna(global_target_mean)
                    print(f"Categorical column '{col}': TargetEncoded using stored map. Unseen filled with global mean {global_target_mean:.4f}.")
                else:
                    X[col] = global_target_mean # Fallback if map not found
                    print(f"Warning: Target mean map for '{col}' not found. Values set to global mean {global_target_mean:.4f}.")
            # Ensure no NaNs remain after mapping (e.g., if a category was all NaNs during training for y)
            X[col] = X[col].fillna(global_target_mean if is_training else imputation_values['numerical'].get(col, global_target_mean))
        else:
            print(f"Warning: Column '{col}' for target encoding not found in X.")

    # Frequency Encoding for any other specified columns (if any)
    if cols_for_freq_encoding: # Only run if there are columns specified
        print("\n--- Frequency Encoding ---")
        for col in cols_for_freq_encoding:
            if col in X.columns: # Should be a categorical column not already target encoded
                if is_training:
                    freq_map = X[col].value_counts(normalize=True) 
                    frequency_maps[col] = freq_map
                    X[col] = X[col].map(freq_map)
                    print(f"Categorical column '{col}': FrequencyEncoded.")
                else:
                    if col in frequency_maps:
                        freq_map = frequency_maps[col]
                        X[col] = X[col].map(freq_map).fillna(0) 
                        print(f"Categorical column '{col}': FrequencyEncoded using stored map.")
                    else:
                        X[col] = 0 
                        print(f"Warning: Frequency map for '{col}' not found. Values set to 0.")
                X[col] = X[col].fillna(0) 
            else:
                print(f"Warning: Column '{col}' for frequency encoding not found in X.")

    # Re-identify numerical and categorical columns AFTER all encoding that converts types
    numerical_cols = X.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist() 

    print(f"\nIdentified Numerical Columns (post-all-encoding): {numerical_cols}")
    print(f"Identified Categorical Columns (post-all-encoding): {categorical_cols}")

    # Imputation (Round 2 - For any new NaNs in encoded cols, or remaining numerical)
    # And for any numerical columns that might have been missed or became NaN due to other ops.
    # This is more of a safeguard.
    print("\n--- Imputation (Round 2 - Post Freq Encoding Safeguard) ---")
    for col in numerical_cols:
        if X[col].isnull().any(): # Only impute if there are still NaNs
            if is_training: # Should not happen if imputation round 1 was complete
                median_val = X[col].median() 
                imputation_values['numerical'][col] = median_val # Overwrite if necessary
                X[col] = X[col].fillna(median_val)
                print(f"Safeguard Imputation: Numerical column '{col}': Imputed NaNs with median ({median_val:.2f}).")
            else:
                median_val = imputation_values['numerical'].get(col, 0)
                X[col] = X[col].fillna(median_val)
                print(f"Safeguard Imputation: Numerical column '{col}': Imputed NaNs with stored/default median.")


    # Label Encoding for REMAINING categorical features
    print("\n--- Label Encoding (for remaining categorical) ---")
    for col in categorical_cols: # These are categorical_cols identified *after* freq encoding
        if col in X.columns:
            if is_training:
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
                label_encoders[col] = le
                print(f"Categorical column '{col}': LabelEncoded.")
            else:
                if col in label_encoders:
                    le = label_encoders[col]
                    X[col] = X[col].astype(str).apply(lambda x: le.transform([x])[0] if x in le.classes_ else -1) 
                else:
                    X[col] = -1 
                    print(f"Warning: LabelEncoder for '{col}' not found. Values set to -1.")
    
    # Scaling Numerical Features (includes frequency encoded columns now)
    print("\n--- Scaling Numerical Features ---")
    if numerical_cols:
        valid_numerical_cols = [nc for nc in numerical_cols if nc in X.columns]
        if valid_numerical_cols:
            if is_training:
                X[valid_numerical_cols] = scaler.fit_transform(X[valid_numerical_cols])
            else:
                if hasattr(scaler, 'mean_'): 
                    X[valid_numerical_cols] = scaler.transform(X[valid_numerical_cols])
                else:
                    print("Warning: Scaler not fitted. Numerical features not scaled for prediction.")
                    # X[valid_numerical_cols] = X[valid_numerical_cols] # No change if scaler not fit
            print(f"Numerical columns scaled: {valid_numerical_cols}")
        else:
            print("No valid numerical columns found to scale.")
    else:
        print("No numerical columns to scale.")

    # PCA section removed. X is now the final set of features.

    # --- Feature Selection based on SHAP (Top 20 features) ---
    # This list is based on the SHAP plot provided earlier.
    top_20_features_from_shap = [
        'Chức danh', 'Ngày bắt đầu HĐ_Month', 'Ngày bắt đầu HĐ_Day', 'Địa Điểm Làm Việc', 
        'Ngày bắt đầu HĐ_Year', 'Tổng Mức Lương', 'Mức thưởng KPI hoặc Khoán', 'Thâm niên',
        'Loại hợp đồng', 'Lương BHXH', 'Phòng Ban', 'Mức Phụ cấp', 
        'Lương KPI - Khoán', 'Ngày vào làm_Year', 'Tổ / Bộ phận', 
        'Ngày kết thúc thử việc_Year', 'Công Ty', 'Chuyên ngành', 'Độ tuổi', 'Nhóm công ty'
    ]
    
    # This variable will hold the list of features to actually use,
    # either top_20_features_from_shap during training, or loaded during prediction.
    selected_features_to_use = None

    if is_training:
        # Ensure all selected features actually exist in X after preprocessing
        final_selected_features = [col for col in top_20_features_from_shap if col in X.columns]
        if len(final_selected_features) != len(top_20_features_from_shap):
            print(f"Warning: Some SHAP-selected features were not found in the processed X. Using {len(final_selected_features)} available features.")
            missing_in_X = [col for col in top_20_features_from_shap if col not in X.columns]
            print(f"Features from SHAP list missing in X: {missing_in_X}")
        
        # If some of the top 20 are missing, we proceed with those that are available from the list.
        # It's crucial that the columns in `final_selected_features` exist in `X`.
        if not all(item in X.columns for item in final_selected_features):
            print("Critical Error: Not all `final_selected_features` are in X.columns. Aborting feature selection.")
            # Fallback to using all columns if there's a major mismatch, or handle error appropriately
            feature_names_out = X.columns.tolist() # Fallback
        else:
            X = X[final_selected_features]
            feature_names_out = final_selected_features
        selected_features_to_use = feature_names_out # This list is saved
        print(f"\nApplied SHAP-based Feature Selection. Selected {len(feature_names_out)} features: {feature_names_out}")

    else: # Prediction time
        if 'selected_features_after_shap' in saved_objects:
            selected_features_to_use = saved_objects['selected_features_after_shap']
            # Ensure X (which was reordered based on 'feature_names_out' from training)
            # now gets filtered by these selected_features_after_shap.
            # The 'feature_names_out' loaded earlier might be the full list *before* SHAP selection.
            # So, we re-filter X based on the *actual* selected list.
            
            # Check if all selected_features_to_use are present in current X.columns
            # (X should have been reordered to expected_feature_names from training,
            # which should be the feature_names_out *after* SHAP selection if saved correctly)
            current_X_cols = X.columns.tolist()
            final_cols_for_prediction = [col for col in selected_features_to_use if col in current_X_cols]

            if len(final_cols_for_prediction) != len(selected_features_to_use):
                 print(f"Warning: Mismatch in selected features for prediction. Expected {len(selected_features_to_use)}, found {len(final_cols_for_prediction)} in current X.")
                 # This case needs careful handling. For now, use what's available.
            
            if not final_cols_for_prediction: # If list is empty, something is very wrong
                print("Error: No selected features available for prediction. Using all columns of X as fallback.")
                feature_names_out = current_X_cols
            else:
                X = X[final_cols_for_prediction]
                feature_names_out = final_cols_for_prediction
            print(f"\nApplied SHAP-based Feature Selection for prediction. Using {len(feature_names_out)} features.")
        else:
            print("Warning: 'selected_features_after_shap' not found in saved preprocessor objects. Using all available features from X.")
            feature_names_out = X.columns.tolist() # Fallback to all columns of X

    print(f"\nPreprocessing complete. Final feature shape: {X.shape}")
    print(f"Final feature names: {feature_names_out}")

    if is_training:
        # Save preprocessor objects
        if preprocessor_objects_path:
            try:
                preprocessor_to_save = {
                    'label_encoders': label_encoders, 
                    'frequency_maps': frequency_maps, 
                    'target_mean_maps': target_mean_maps, 
                    'global_target_mean': global_target_mean, 
                    'scaler': scaler,
                    'imputation_values': imputation_values,
                    'numerical_cols': numerical_cols, # These are numerical *before* this SHAP selection
                    'categorical_cols': categorical_cols, # These are categorical *before* this SHAP selection
                    'feature_names_out': feature_names_out, # This will be the selected feature list
                    'selected_features_after_shap': selected_features_to_use if is_training else saved_objects.get('selected_features_after_shap') # Save/ensure it's there
                }
                joblib.dump(preprocessor_to_save, preprocessor_objects_path)
                print(f"Preprocessor objects saved to {preprocessor_objects_path}")
            except Exception as e:
                print(f"Error saving preprocessor objects: {e}")
        return X, y, feature_names_out # X is now the final dataframe
    else:
        return X, feature_names_out # X is now the final dataframe


if __name__ == '__main__':
    # Example usage (for testing the preprocessing script itself)
    DATA_FILE_PATH_EXAMPLE = r"C:\Users\Dell03.LAPTOP-PTNNLMOI\Desktop\Báo cáo HR\Danh sách NV\12052025 - Danh sách nhân sự KTG.xlsx"
    PREPROCESSOR_SAVE_PATH_EXAMPLE = '../models/test_preprocessor.joblib' # Save in models directory relative to src

    # Test training preprocessing
    print("--- TESTING TRAINING PREPROCESSING ---")
    raw_df_train = load_data(DATA_FILE_PATH_EXAMPLE)
    if raw_df_train is not None:
        X_train_processed, y_train_processed, features_train = preprocess_data(raw_df_train.copy(), is_training=True, preprocessor_objects_path=PREPROCESSOR_SAVE_PATH_EXAMPLE)
        if X_train_processed is not None:
            print("\nTraining preprocessing successful.")
            print("X_train_processed shape:", X_train_processed.shape)
            print("y_train_processed shape:", y_train_processed.shape)
            print("First 5 rows of X_train_processed:\n", X_train_processed.head())

    # Test prediction preprocessing (using a sample of the same data for simplicity)
    print("\n--- TESTING PREDICTION PREPROCESSING ---")
    if raw_df_train is not None:
        # Simulate prediction data (e.g., first 10 rows, without target)
        # In a real scenario, this would be new, unseen data.
        df_pred_sample = raw_df_train.head(10).drop(columns=['Đang làm việc'], errors='ignore')
        
        X_pred_processed, features_pred = preprocess_data(df_pred_sample.copy(), is_training=False, preprocessor_objects_path=PREPROCESSOR_SAVE_PATH_EXAMPLE)
        if X_pred_processed is not None:
            print("\nPrediction preprocessing successful.")
            print("X_pred_processed shape:", X_pred_processed.shape)
            print("First 5 rows of X_pred_processed:\n", X_pred_processed.head())
            # Check if columns match
            if features_train and features_pred and features_train == features_pred:
                print("Feature names match between training and prediction preprocessing.")
            else:
                print("Warning: Feature names MISMATCH between training and prediction preprocessing.")
                print("Training features:", features_train)
                print("Prediction features:", features_pred)
