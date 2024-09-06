import time
import logging
from solver.platform.hems import HomeEnergyManagementSystem
from solver.methods.data_loader import load_data
from solver.methods.run_scenario import run_scenario
from solver.methods.dataset_aggregation import dataset_aggregation

from utils.common_utils import save_results, save_dataset

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# def get_train_data():
#     return load_data(data_start_time = '2012-01-01 12:00:00', data_end_time = '2014-02-01 11:00:00')

def main():
    # Timing the execution
    start_time = time.time()

    # Load data
    data = load_data()

    # Initialize Microgrid
    hems = HomeEnergyManagementSystem()

    # Define the number of scenarios
    num_scenarios = len(data['initial_soc_ev'])
    print('num_scenarios:', num_scenarios)

    # Step 1: Process scenarios and store results
    results = run_scenario(hems, data, num_scenarios)

    # Save the results to CSV files
    save_results(results)

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

    # Print elapsed time
    elapsed_time = time.time() - start_time
    logging.info(f"Total elapsed time: {elapsed_time:.2f} seconds")

#python3 src/data_generation.py
if __name__ == "__main__":
    main()