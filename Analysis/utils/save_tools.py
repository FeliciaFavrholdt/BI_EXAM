import os
import json
from datetime import datetime
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from IPython.display import display  # We implementet this, so we could display the plots/models, both places (folder and notebooks) 

# -------------------- Save Notebook Summaries --------------------
def save_notebook_and_summary(notebook_name, summary, folder_path="../reports"):
    """
    Saves a structured project summary with metadata to a JSON file in the reports folder.
    
    Parameters:
    - notebook_name (str): The name of the notebook being summarized (used in the filename).
    - summary (dict): A dictionary with metadata such as description, team_members, etc.
    - folder_path (str): Path to the reports folder (default is '../reports')
    """
    # Create folder if it doesn't exist
    os.makedirs(folder_path, exist_ok=True)

    # Timestamped filename
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"{notebook_name}_summary_{timestamp}.json"
    filepath = os.path.join(folder_path, filename)

    # Save summary as JSON
    with open(filepath, "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=4)

    print(f"Summary saved to: {filepath}")


# -------------------- Saving Plots --------------------
def save_plot(fig: Figure, filename: str, caption: str = "", folder_path: str = "../plots"):
    """
    Saves a matplotlib/seaborn figure and its caption for Streamlit use.

    Parameters:
    - fig (Figure): The matplotlib/seaborn figure object
    - filename (str): The image filename (e.g., 'age_distribution.png')
    - caption (str): Optional caption to save alongside the image
    - folder_path (str): Folder to save the image and caption (default: '../plots')

    Also saves caption as a .txt file with same base name.
    """
    os.makedirs(folder_path, exist_ok=True)

    # Show the figure inside the notebook
    display(fig)

    # Save image
    image_path = os.path.join(folder_path, filename)
    fig.savefig(image_path, dpi=300, bbox_inches="tight")

    # Save caption
    if caption:
        caption_filename = filename.rsplit(".", 1)[0] + ".txt"
        caption_path = os.path.join(folder_path, caption_filename)
        with open(caption_path, "w", encoding="utf-8") as f:
            f.write(caption)

    # Close figure after saving
    plt.close(fig)

    print(f"Plot saved to: {image_path}")
    if caption:
        print(f"Caption saved to: {caption_path}")


# -------------------- Saving Model Results --------------------
def save_model_results(model_results, filename="model_results.json", folder_path="../data"):
    """
    Saves model evaluation results to a JSON file.

    Parameters:
    - model_results (list): A list of dicts where each dict contains model name, accuracy, AUC, etc.
    - filename (str): Name of the JSON file (default: 'model_results.json')
    - folder_path (str): Folder to save the file (default: '../data')
    """
    os.makedirs(folder_path, exist_ok=True)
    
    filepath = os.path.join(folder_path, filename)

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(model_results, f, indent=4)

    print(f"Model results saved to: {filepath}")
