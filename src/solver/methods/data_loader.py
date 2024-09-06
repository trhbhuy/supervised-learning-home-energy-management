import os
import pandas as pd
import numpy as np
from typing import Dict, Optional
from .. import config as cfg

def load_data(
    is_train: Optional[bool] = None,
    data_start_time: Optional[str] = None,
    data_end_time: Optional[str] = None,
    file_path: str = os.path.join(cfg.RAW_DATA_DIR, 'historical_data.csv'),
    ev_file_path: str = os.path.join(cfg.RAW_DATA_DIR, 'ev_data.csv')
) -> Dict[str, np.ndarray]:
    """Load the dataset into specific data sequences with defined time ranges.
    
    Args:
        is_train (Optional[bool]): If True, load training data. If False, load testing data. If None, load all data.
        data_start_time (Optional[str]): Start time for filtering the data. Overrides is_train if provided.
        data_end_time (Optional[str]): End time for filtering the data. Overrides is_train if provided.
        file_path (str): Path to the CSV file containing the data.
        ev_file_path (str): Path to the CSV file containing the ev data.

    Returns:
        Dict[str, np.ndarray]: 
            - 'rtp': Real-time pricing (array)
            - 'p_pv_max': Solar power (array)
            - 'p_if': Inflexible load (array)
            - 'p_fl_1': Flexible load 1 (array)
            - 'p_fl_2': Flexible load 2 (array)
    """
    # Load the data
    dataframe = pd.read_csv(file_path, index_col='DateTime', parse_dates=True)
    ev_dataframe = pd.read_csv(ev_file_path, index_col='DateTime', parse_dates=True)

    # Determine the time range based on data_start_time, data_end_time, and is_train
    if data_start_time and data_end_time:
        # Use the provided custom time range
        data = dataframe.loc[data_start_time:data_end_time]
        data_ev = ev_dataframe.loc[data_start_time:data_end_time]
    elif is_train is None:
        # Consider the entire dataframe
        data = dataframe
        data_ev = ev_dataframe
    elif is_train:
        # Load training data
        data_start_time, data_end_time = '2012-01-01 12:00:00', '2014-02-01 11:00:00'
        data = dataframe.loc[data_start_time:data_end_time]

        data_ev_start_time, data_ev_end_time = '2012-01-01', '2014-01-31'
        data_ev = ev_dataframe.loc[data_ev_start_time:data_ev_end_time]
    else:
        # Load testing data
        data_start_time, data_end_time = '2014-02-01 12:00:00', '2014-02-27 11:00:00'
        data = dataframe.loc[data_start_time:data_end_time]

        data_ev_start_time, data_ev_end_time = '2014-02-01', '2014-02-26'
        data_ev = ev_dataframe.loc[data_ev_start_time:data_ev_end_time]

    # Store the data in a dictionary
    output_data = {
        'rtp': data['RTP'].values,
        'p_pv_max': data['PV Power'].values,
        'p_if': data['Load'].values,
        'initial_soc_ev': data_ev['Initial SOC'].values,
        't_ev_arrive': data_ev['ArriveTime'].values,
        't_ev_depart': data_ev['DepartureTime'].values,
    }
    
    return output_data