import os
import pandas as pd
import numpy as np 
from flask import Flask, request, render_template, redirect, url_for, flash, send_from_directory # Added send_from_directory
import joblib
from werkzeug.utils import secure_filename
# Assuming preprocessing and train_model modules are structured to allow import
# For example, by being in the same directory or by adjusting PYTHONPATH
from preprocessing import load_data, preprocess_data # Corrected import
# train_model functions might not be directly used by app.py if model is pre-trained

# Define paths - ensure these are relative to the app.py location or absolute
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'uploads') # Ensure UPLOAD_FOLDER is absolute or relative to app.py
DOWNLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'downloads') # For storing files to be downloaded
ALLOWED_EXTENSIONS = {'xlsx'}

# Updated paths to match train_model.py
MODEL_DIR_APP = os.path.join(os.path.dirname(__file__), '..', 'models') # Define model directory relative to app.py
MODEL_PATH = os.path.join(MODEL_DIR_APP, 'best_attrition_model.joblib')
PREPROCESSOR_ARTIFACTS_PATH = os.path.join(MODEL_DIR_APP, 'preprocessor_artifacts.joblib')


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['DOWNLOAD_FOLDER'] = DOWNLOAD_FOLDER # Add DOWNLOAD_FOLDER to app.config
app.secret_key = "supersecretkey" # Needed for flash messages

# Ensure upload and download folders exist
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
if not os.path.exists(DOWNLOAD_FOLDER):
    os.makedirs(DOWNLOAD_FOLDER)

# Load the trained model and preprocessor objects
model = None
preprocessor_artifacts = None # Renamed for clarity
try:
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
        print("Model loaded successfully.")
    else:
        print(f"Warning: Model not found at {MODEL_PATH}. Prediction will not work.")

    if os.path.exists(PREPROCESSOR_ARTIFACTS_PATH):
        preprocessor_artifacts = joblib.load(PREPROCESSOR_ARTIFACTS_PATH)
        print("Preprocessor artifacts loaded successfully.")
    else:
        print(f"Warning: Preprocessor artifacts not found at {PREPROCESSOR_ARTIFACTS_PATH}. Prediction may be unreliable.")
        
except Exception as e:
    print(f"Error loading model or preprocessor artifacts: {e}")
    if model is None: print("Model loading failed.")
    if preprocessor_artifacts is None: print("Preprocessor artifacts loading failed.")


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def upload_file_and_predict():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            flash('File successfully uploaded')

            if model is None or preprocessor_artifacts is None:
                flash('Model or preprocessor artifacts not loaded. Cannot make predictions.')
                return render_template('index.html', filename=filename, predictions_html=None)

            try:
                # Load the uploaded data
                new_data_df_raw = load_data(filepath)
                if new_data_df_raw is None:
                    flash('Could not load data from the uploaded file.')
                    return render_template('index.html', filename=filename, predictions_html=None, download_filename=None)

                # Initialize the DataFrame for download with all original columns
                df_for_download = new_data_df_raw.copy()
                df_for_download['AttritionPrediction'] = ''  # Default to blank
                df_for_download['AttritionProbability (%)'] = ''  # Default to blank

                # Determine which rows to make predictions for
                df_to_predict_on = new_data_df_raw.copy() # Start with all data
                indices_to_predict_on = df_to_predict_on.index # By default, all original indices

                if 'Đang làm việc' in df_to_predict_on.columns:
                    # Ensure 'Đang làm việc' is numeric for reliable comparison (1 means working)
                    df_to_predict_on['Đang làm việc'] = pd.to_numeric(df_to_predict_on['Đang làm việc'], errors='coerce')
                    working_mask = (df_to_predict_on['Đang làm việc'] == 1)
                    
                    if not working_mask.any():
                        flash('No employees found with "Đang làm việc" status as 1. No predictions generated.')
                        df_to_predict_on = pd.DataFrame() # Empty df, so no predictions run
                    else:
                        df_to_predict_on = df_to_predict_on[working_mask]
                        indices_to_predict_on = df_to_predict_on.index # Update to original indices of working employees
                
                if df_to_predict_on.empty:
                    # This handles cases where the file was empty, or no one was "Đang làm việc == 1"
                    if not ('Đang làm việc' in new_data_df_raw.columns and not working_mask.any()): # Avoid double flashing
                         flash('No data available for prediction after filtering.')
                else:
                    # Preprocess the subset of data that needs prediction
                    # preprocess_data will handle dropping 'Đang làm việc' if it's still in df_to_predict_on
                    X_processed_subset, _ = preprocess_data(
                        df_to_predict_on.copy(),  # Pass a copy of the filtered DataFrame
                        is_training=False,
                        preprocessor_objects_path=PREPROCESSOR_ARTIFACTS_PATH
                    )

                    if X_processed_subset is None:
                        flash('Error during data preprocessing for prediction. Check server logs.')
                        # df_for_download will have blank predictions
                    elif X_processed_subset.empty:
                        flash('No data remained after preprocessing the selected rows. No predictions generated.')
                        # df_for_download will have blank predictions
                    else:
                        # Ensure all data in X_processed_subset is numeric before prediction
                        non_numeric_cols_pred = X_processed_subset.select_dtypes(exclude=np.number).columns
                        if not non_numeric_cols_pred.empty:
                            flash(f"Warning: Non-numeric columns found after prediction preprocessing: {non_numeric_cols_pred.tolist()}. Attempting to coerce.")
                            for col in non_numeric_cols_pred:
                                X_processed_subset[col] = pd.to_numeric(X_processed_subset[col], errors='coerce').fillna(0)
                        
                        if X_processed_subset.isnull().sum().sum() > 0:
                            flash("Warning: NaNs found in data before prediction. Filling with 0. Check preprocessing logic.")
                            X_processed_subset = X_processed_subset.fillna(0)
                        
                        # Make predictions on the processed subset
                        predictions_subset = model.predict(X_processed_subset)
                        probabilities_subset = model.predict_proba(X_processed_subset)[:, 1] # Probability of attrition

                        # Update the corresponding rows in the main df_for_download DataFrame
                        # using the original indices stored in indices_to_predict_on
                        for i, original_idx in enumerate(indices_to_predict_on):
                            df_for_download.loc[original_idx, 'AttritionPrediction'] = 'Yes (Will Leave)' if predictions_subset[i] == 1 else 'No (Will Stay)'
                            df_for_download.loc[original_idx, 'AttritionProbability (%)'] = f"{probabilities_subset[i]*100:.2f}"
                
                # Prepare data for HTML display (can be a subset of columns from df_for_download)
                display_cols_for_html = []
                # Add some identifying columns if they exist
                if 'Họ Và Tên' in df_for_download.columns: display_cols_for_html.append('Họ Và Tên')
                if 'Mã Nhân Viên' in df_for_download.columns: display_cols_for_html.append('Mã Nhân Viên')
                if 'Đang làm việc' in df_for_download.columns: display_cols_for_html.append('Đang làm việc') # Show original status
                
                # Add prediction columns
                display_cols_for_html.extend(['AttritionPrediction', 'AttritionProbability (%)'])
                
                # Filter out columns that might not exist in df_for_download (e.g., if input was minimal)
                actual_display_cols = [col for col in display_cols_for_html if col in df_for_download.columns]
                
                predictions_html = df_for_download[actual_display_cols].to_html(classes='table table-striped table-sm', index=False, border=0)

                # Save the full df_for_download to an Excel file
                download_filename = f"predictions_{secure_filename(filename)}" # Use secure_filename for safety
                download_filepath = os.path.join(app.config['DOWNLOAD_FOLDER'], download_filename)
                
                # Debugging: Print info about df_for_download before saving
                print(f"--- Debug: df_for_download (before saving to Excel) ---")
                print(f"Shape: {df_for_download.shape}")
                print(f"Columns: {df_for_download.columns.tolist()}")
                print(f"Data types:\n{df_for_download.dtypes}")
                if df_for_download.empty:
                    print("df_for_download is EMPTY!")
                else:
                    print(f"Head:\n{df_for_download.head()}")
                print(f"--- End Debug ---")

                df_for_download.to_excel(download_filepath, index=False, engine='openpyxl')
                
                return render_template('index.html', filename=filename, predictions_html=predictions_html, download_filename=download_filename)

            except Exception as e:
                flash(f'Error processing file or making predictions: {str(e)}')
                print(f"Error during prediction: {str(e)}")
                import traceback
                traceback.print_exc() # Print full traceback for debugging
                return render_template('index.html', filename=filename, predictions_html=None)
        else:
            flash('Allowed file type is .xlsx')
            return redirect(request.url)

    return render_template('index.html', predictions_html=None, download_filename=None)

@app.route('/download/<filename>')
def download_file(filename):
    return send_from_directory(app.config['DOWNLOAD_FOLDER'], filename, as_attachment=True)

if __name__ == '__main__':
    # Make sure to run this from the 'src' directory or adjust paths accordingly
    # For development:
    # cd employee_attrition_predictor/src
    # python app.py
    app.run(debug=True)
