import pandas as pd

# Path to the dataset
DATA_FILE_PATH = r"C:\Users\Dell03.LAPTOP-PTNNLMOI\Desktop\employee_attrition_predictor\data\Danh sách nhân sự.xlsx"

def inspect_dataset_columns(file_path):
    """Loads the dataset and prints column names and their inferred data types."""
    try:
        df = pd.read_excel(file_path, engine='openpyxl')
        print("Dataset Columns and Data Types:")
        print("-------------------------------")
        for column in df.columns:
            print(f"Column: {column}, Dtype: {df[column].dtype}")
        print("\n-------------------------------")
        print("First 5 rows of the dataset:")
        print(df.head())
        print("\n-------------------------------")
        print("Basic info:")
        df.info()
        print("\n-------------------------------")
        print("Missing values per column:")
        print(df.isnull().sum())

    except FileNotFoundError:
        print(f"Error: The file {file_path} was not found.")
    except Exception as e:
        print(f"Error loading or inspecting data: {e}")

if __name__ == '__main__':
    inspect_dataset_columns(DATA_FILE_PATH)
