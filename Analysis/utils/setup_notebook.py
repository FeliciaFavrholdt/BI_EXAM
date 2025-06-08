import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from IPython.display import display
from sklearn.preprocessing import StandardScaler

def init_environment():
    """
    Sets up folders and styling for reproducible project execution.
    Creates the following folders if they don't exist:
    - ../data
    - ../models
    - ../plots
    - ../reports
    """
    sns.set_theme(style="whitegrid", palette="deep")
    pd.set_option('display.max_columns', None)

    folders = ["../data", "../models", "../plots", "../reports"]
    for folder in folders:
        os.makedirs(folder, exist_ok=True)

    print("Environment setup complete.")

def load_csv(filepath):
    """
    Loads a CSV file and returns a DataFrame.
    """
    df = pd.read_csv(filepath)
    print(f"Loaded data from {filepath} with shape {df.shape}")
    return df 

def quick_overview(df, preview_columns=None):
    """
    Provides a structured overview of the dataset.
    Includes shape, data types, basic statistics, and a sample of the data.
    
    Parameters:
    - df (DataFrame): The dataset to inspect.
    - preview_columns (int or None): If set, limits the number of columns shown in the statistical summary.
    """
    print("----- Dataset Shape -----")
    print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
    
    print("\n----- Data Types and Non-Null Counts -----")
    df.info(verbose=False, show_counts=True)

    print("\n----- Overview of dataset -----")
    df.info()

    print("\n----- Statistical Summary -----")
    print("This summary includes counts, mean, standard deviation, min, max, and percentiles for numeric and categorical columns.\n")
    if preview_columns:
        display(df.describe(include="all").iloc[:, :preview_columns])
    else:
        display(df.describe(include="all"))

    print("\n----- First 5 Rows -----")
    display(df.head())
