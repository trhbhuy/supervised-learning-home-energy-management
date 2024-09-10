import os
import pandas as pd
import pickle
import logging
from typing import Dict
from solver.config import PROCESSED_DATA_DIR, GENERATED_DATA_DIR

def save_pickle(obj, file_path: str):
    with open(file_path, 'wb') as file:
        pickle.dump(obj, file)

def load_pickle(file_path: str):      
    # Load the label scaler
    with open(file_path, 'rb') as f:
        return pickle.load(f)
    
def save_results(results: dict, output_dir: str = PROCESSED_DATA_DIR):
    """Save optimization results to CSV files.

    Args:
        results (dict): The dictionary containing optimization results.
        output_dir (str): Directory to save the CSV files.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for key, value in results.items():
        output_path = os.path.join(output_dir, f"{key}.csv")
        if value.ndim > 2:
            value = value.reshape(value.shape[0], -1)
        pd.DataFrame(value).to_csv(output_path, index=False)
        logging.info(f"Saved {key} to {output_path}")

def save_dataset(dataset: dict, filename: str, subdirectory: str = ''):
    """Save the dataset to a file using pickle, with an optional subdirectory.

    Args:
        dataset (dict): The dataset containing X_data and y_data.
        filename (str): The filename to save the dataset to.
        subdirectory (str, optional): Additional folder to save the dataset in. Defaults to None.
    """
    # Construct the full path with optional subdirectory
    file_path = os.path.join(GENERATED_DATA_DIR, subdirectory, filename)

    # Create the subdirectory if it doesn't exist
    dir_path = os.path.dirname(file_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    # Save the dataset as a pickle file
    save_pickle(dataset, file_path)
    logging.info(f"Dataset saved to {file_path}")

def load_results(input_dir: str) -> dict:
    """Load optimization results from CSV files.

    Args:
        input_dir (str): Directory from which to load the CSV files.

    Returns:
        dict: A dictionary containing the loaded results.
    """
    results = {}
    
    # Check if the input directory exists
    if not os.path.exists(input_dir):
        raise FileNotFoundError(f"The directory {input_dir} does not exist.")
    
    # Iterate over all CSV files in the directory
    for file_name in os.listdir(input_dir):
        if file_name.endswith(".csv"):
            # Extract the key from the file name (removing the .csv extension)
            key = file_name.replace(".csv", "")
            
            # Load the CSV file into a DataFrame, and then convert it to a NumPy array
            file_path = os.path.join(input_dir, file_name)
            df = pd.read_csv(file_path)
            
            # Convert DataFrame back to NumPy array
            results[key] = df.to_numpy()
            logging.info(f"Loaded {key} from {file_path}")
    
    return results
