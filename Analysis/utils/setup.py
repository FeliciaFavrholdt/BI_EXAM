import os
import seaborn as sns
import matplotlib.pyplot as plt

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

    folders = ["../data", "../models", "../plots", "../reports"]
    for folder in folders:
        os.makedirs(folder, exist_ok=True)

    print("Environment setup complete.")
