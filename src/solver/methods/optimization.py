import logging
import numpy as np
from typing import Dict
from .feature_engineering import feature_engineering

def initialize_records() -> Dict[str, list]:
    """Initialize a dictionary of lists to store results for multiple scenarios."""
    return {key: [] for key in [
        'ObjVal', 'p_grid_pur', 'p_grid_exp', 'u_grid_pur', 'u_grid_exp', 'p_pv',
        'p_ess_ch', 'p_ess_dch', 'u_ess_ch', 'u_ess_dch', 'soc_ess',
        'p_ev_ch', 'p_ev_dch', 'u_ev_ch', 'u_ev_dch', 'soc_ev', 'ev_time_range',
        'rtp', 'p_pv_max', 'p_if'
    ]}

def process_single_scenario(microgrid, data, start_idx, end_idx, scn) -> Dict[str, np.ndarray]:
    """Run optimization for a single scenario and return the results."""
    try:
        return microgrid.optim(
            data['rtp'][start_idx:end_idx],
            data['p_pv_max'][start_idx:end_idx],
            data['p_if'][start_idx:end_idx],
            data['t_ev_arrive'][scn],
            data['t_ev_depart'][scn],
            data['soc_ev_init'][scn]
        )
    except Exception as e:
        logging.error(f"Error in scenario {scn+1}: {str(e)}")
        raise

def run_optim(microgrid: object, data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Process optimization for multiple scenarios and store results in a dictionary.

    Args:
        microgrid (object): The microgrid object to perform optimization.
        data (Dict[str, np.ndarray]): Solar, wind, and load data.

    Returns:
        Dict[str, np.ndarray]: A dictionary containing results for all scenarios.
    """
    # Initialize dictionary to store results
    records = initialize_records()

    # Define the number of time steps and scenarios
    T_num = microgrid.T_num
    num_scenarios = len(data['rtp']) // T_num
    logging.info(f"Processing {num_scenarios} scenarios...")

    # Process each scenario
    for scn in range(num_scenarios):
        start_idx = scn * T_num
        end_idx = start_idx + T_num
        
        # Process the scenario and retrieve results
        try:
            results = process_single_scenario(microgrid, data, start_idx, end_idx, scn)
        except Exception:
            continue  # If there's an error, skip to the next scenario

        # Append each result to the corresponding list in records
        for key in records.keys():
            records[key].append(results[key])

        logging.info(f"Scenario {scn+1}: Optimal ObjVal = {results['ObjVal']:.4f}")

    # Convert lists to numpy arrays for final storage
    logging.info("Converting records to numpy arrays...")
    records = {key: np.array(val) for key, val in records.items()}

    # Apply feature engineering and calculate additional features
    logging.info("Applying feature engineering...")
    records = feature_engineering(microgrid, records)

    logging.info("Scenario processing completed.")
    return records