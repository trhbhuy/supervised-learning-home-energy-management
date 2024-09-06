import os
import logging
from solver.platform.hems import HomeEnergyManagementSystem
from solver.methods.data_loader import load_data
from solver.methods.run_scenario import run_scenario
from solver.methods.dataset_aggregation import dataset_aggregation

from utils.common_utils import load_results, save_dataset

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define base paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

def main():
    # Save the results to CSV files
    input_dir = os.path.join(BASE_DIR, 'data', 'processed')
    results = load_results(input_dir)

    # Step 2: Process the dataset to obtain training dataset
    dataset = dataset_aggregation(results, feature_keys=['time_step', 'rtp', 'p_net', 'soc_ess_prev', 'soc_ev_prev'], label_keys=['p_ess', 'p_ev'])
    ess_dataset = dataset_aggregation(results, feature_keys=['time_step', 'rtp', 'p_net', 'soc_ess_prev'], label_keys=['p_ess'])
    ev_dataset = dataset_aggregation(results, feature_keys=['time_step', 'rtp', 'p_net', 'soc_ev_prev'], label_keys=['p_ev'])

    # Log progress
    logging.info(f"data_seq shape: {dataset['data_seq'].shape}, label shape: {dataset['label'].shape}")
    logging.info(f"ess data_seq shape: {ess_dataset['data_seq'].shape}, label shape: {ess_dataset['label'].shape}")
    logging.info(f"ev data_seq shape: {ev_dataset['data_seq'].shape}, label shape: {ev_dataset['label'].shape}")

    # Save the dataset
    save_dataset(dataset, "dataset.pkl")
    save_dataset(ess_dataset, "dataset.pkl", subdirectory='ess')
    save_dataset(ev_dataset, "dataset.pkl", subdirectory='ev')

#python3 src/aggregation.py
if __name__ == "__main__":
    main()