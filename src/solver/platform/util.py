import os
import numpy as np
from typing import Dict, Optional

from ..config import GENERATED_DATA_DIR
from ..utils.file_util import load_pickle
from ..utils.numeric_util import get_boundary_tolerance, get_deviation

def scaler_loader(subdir=''):
    """Load the data and label scalers from the generated files."""
    # Paths to the scaler files
    data_scaler_path = os.path.join(GENERATED_DATA_DIR, subdir, 'data_scaler.pkl')
    label_scaler_path = os.path.join(GENERATED_DATA_DIR, subdir, 'label_scaler.pkl')

    # Load the data scaler
    data_scaler = load_pickle(data_scaler_path)

    # Load the label scaler
    label_scaler = load_pickle(label_scaler_path)
            
    return data_scaler, label_scaler

def check_boundary_constraint(value, lower_limit, upper_limit):
    """
    Calculate the boundary violation if the value is outside the specified limits.

    Args:
        value (float): The value to check.
        lower_limit (float): The minimum allowable value.
        upper_limit (float): The maximum allowable value.

    Returns:
        float: The calculated boundary violation if the value is out of bounds, otherwise 0.0.
    """
    return get_boundary_tolerance(value, lower_limit, upper_limit)

def check_ramp_constraint(new_value, prior_value, ramp_rate):
    """
    Check if the new value violates the ramp rate constraint based on the prior value.

    Args:
        new_value (float): The value to check.
        prior_value (float): The value from the prior timestep.
        ramp_rate (float): The maximum allowable change between successive values.

    Returns:
        float: The calculated boundary violation if the ramp rate is exceeded, otherwise 0.0.
    """
    # Calculate the permissible range for the new value based on the ramp rate
    min_allowed_value = prior_value - ramp_rate
    max_allowed_value = prior_value + ramp_rate

    # Check if the new value is within the allowable range
    tolerance = get_boundary_tolerance(new_value, min_allowed_value, max_allowed_value)

    return tolerance

def check_setpoint(value, setpoint, tolerance_threshold=1e-5):
    """
    Calculate the deviation of a value from its setpoint.

    Args:
        value (float): The current value to check.
        setpoint (float): The target setpoint value.
        tolerance_threshold (float, optional): An allowable tolerance range. Defaults to 0.

    Returns:
        float: The deviation from the setpoint. Returns 0 if within the tolerance threshold.
    """
    return get_deviation(value, setpoint, tolerance_threshold)

def get_vars_results(variable, inner_set: np.ndarray, outer_set: Optional[np.ndarray] = None) -> np.ndarray:
    """Extract results from a given variable for the specified sets."""
    if outer_set is not None:
        return np.array([[variable[ii, tt].X for tt in inner_set] for ii in outer_set])
    return np.array([variable[tt].X for tt in inner_set])

def create_results_dict(model, variable_dict, time_set):
    """Create a results dictionary by extracting results from variables."""
    results = {'ObjVal': model.ObjVal}
    
    # Extract results for all variables in the dictionary
    for key, variable in variable_dict.items():
        results[key] = get_vars_results(variable, time_set)
    
    return results