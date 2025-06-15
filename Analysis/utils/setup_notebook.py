import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from IPython.display import display
from sklearn.preprocessing import StandardScaler

# -------------------- Environment Setup --------------------
def init_environment():
    """
    Sets up folders and styling for reproducible project execution.
    """
    sns.set_theme(style="whitegrid", palette="deep")
    pd.set_option('display.max_columns', None)

    folders = ["../data", "../models", "../plots", "../reports"]
    for folder in folders:
        os.makedirs(folder, exist_ok=True)

    print("Environment setup complete.")

# -------------------- Data Loading --------------------
def load_csv(filepath):
    """
    Loads a CSV file and returns a DataFrame.
    """
    df = pd.read_csv(filepath)
    print(f"Loaded data from {filepath} with shape {df.shape}")
    return df 

# -------------------- Dataset Shape --------------------
def print_shape(df):
    """
    Prints the number of rows and columns in the dataset.
    """
    print("----- Dataset Shape -----")
    print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")

# -------------------- Info Summary --------------------
def print_info(df):
    """
    Displays non-null counts and data types.
    """
    print("\n----- Data Types and Non-Null Counts -----")
    df.info(verbose=False, show_counts=True)

def print_full_info(df):
    """
    Displays full info including memory usage and column types.
    """
    print("\n----- Full Dataset Info -----")
    df.info()

# -------------------- Statistical Summary --------------------
def print_description(df, preview_columns=None):
    """
    Displays descriptive statistics of the dataset.
    """
    print("\n----- Statistical Summary -----")
    print("This summary includes count, mean, std, min, max, and percentiles.\n")
    if preview_columns:
        display(df.describe(include="all").T.iloc[:preview_columns, :])
    else:
        display(df.describe(include="all").T)

# -------------------- First Rows --------------------
def show_head(df, n=5):
    """
    Displays the first n rows of the dataset.
    """
    print(f"\n----- First {n} Rows -----")
    display(df.head(n))

# -------------------- Categorical Summary --------------------
def print_categorical_description(df):
    """
    Displays summary statistics for non-numeric (object or category) columns.
    Useful for inspecting categorical variables.
    """
    print("\n----- Categorical Summary -----")
    cat_summary = df.describe(include='object').T
    if cat_summary.empty:
        print("No non-numeric (object) columns to describe.")
    else:
        display(cat_summary)


# -------------------- Load our Models --------------------
def load_model(filepath, model_name="Model"):
    """
    Loads a model from a pickle file using joblib.
    Includes error handling.
    """
    import joblib

    print(f"Loading {model_name} from {filepath}...")
    try:
        model = joblib.load(filepath)
        print(f"{model_name} loaded successfully.")
        return model
    except FileNotFoundError:
        print(f"Error: {model_name} file not found at {filepath}.")
    except Exception as e:
        print(f"Error loading {model_name}: {e}")
    return None
