import time
import logging
from solver.platform.hems import HomeEnergyManagementSystem
from solver.methods.data_loader import load_data
from solver.methods.optimization import run_optim
from solver.methods.dataset_aggregation import dataset_aggregation
from utils.common_utils import save_results, save_dataset

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def process_optimization(hems, data):
    """Run scenarios and aggregate the results."""
    # Run scenarios
    logging.info("Running scenarios...")
    results = run_optim(hems, data)

    # Save results
    logging.info("Saving scenario results...")
    save_results(results)

    return results

def process_datasets(results):
    """Aggregate and save datasets."""
    # Aggregate datasets
    logging.info("Aggregating datasets...")
    dataset = dataset_aggregation(results, feature_keys=['time_step', 'rtp', 'p_net', 'soc_ess_prev', 'soc_ev_prev'], label_keys=['p_ess', 'p_ev'])
    ess_dataset = dataset_aggregation(results, feature_keys=['time_step', 'rtp', 'p_net', 'soc_ess_prev'], label_keys=['p_ess'])
    ev_dataset = dataset_aggregation(results, feature_keys=['time_step', 'rtp', 'p_net', 'soc_ev_prev'], label_keys=['p_ev'])

    # Log dataset shapes
    logging.info(f"Dataset shapes: data_seq: {dataset['data_seq'].shape}, label: {dataset['label'].shape}")
    logging.info(f"ESS dataset shapes: data_seq: {ess_dataset['data_seq'].shape}, label: {ess_dataset['label'].shape}")
    logging.info(f"EV dataset shapes: data_seq: {ev_dataset['data_seq'].shape}, label: {ev_dataset['label'].shape}")

    # Save datasets
    logging.info("Saving aggregated datasets...")
    save_dataset(dataset, "dataset.pkl")
    save_dataset(ess_dataset, "ess_dataset.pkl", subdirectory='ess')
    save_dataset(ev_dataset, "ev_dataset.pkl", subdirectory='ev')

def main():
    """Main entry point for the data generation process."""
    # Timing the execution
    start_time = time.time()

    try:
        # Step 1: Load data
        logging.info("Loading data...")
        data = load_data()
        if data is None:
            raise ValueError("Failed to load data.")

        # Step 2: Initialize the HEMS platform
        logging.info("Initializing Home Energy Management System (HEMS)...")
        hems = HomeEnergyManagementSystem()

        # Step 3: Process optimization process
        results = process_optimization(hems, data)

        # Step 4: Process and save datasets
        process_datasets(results)

    except Exception as e:
        logging.error(f"An error occurred: {e}")
    finally:
        # Log the total elapsed time
        elapsed_time = time.time() - start_time
        logging.info(f"Total elapsed time: {elapsed_time:.2f} seconds")

#python3 src/data_generation.py
# Entry point
if __name__ == "__main__":
    main()