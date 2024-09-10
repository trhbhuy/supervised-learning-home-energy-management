import os
import numpy as np

# Define base paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

# Data directories
DATA_DIR = os.path.join(BASE_DIR, 'data')
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')
GENERATED_DATA_DIR = os.path.join(DATA_DIR, 'generated')

# Time settings
T_NUM = 24  # 24 hours
T_SET = np.arange(T_NUM)
DELTA_T = 24 / T_NUM  # Time step in hours

# Grid exchange parameters
P_GRID_PUR_MAX = 50  # Maximum power purchase from grid (kW)
R_GRID_PUR = 50  # Ramp rate for grid purchase (kW/h)
P_GRID_EXP_MAX = 50  # Maximum power export to grid (kW)
R_GRID_EXP = 50  # Ramp rate for grid export (kW/h)
PHI_RTP = 1  # Real-time pricing factor

# Solar PV parameters
P_PV_RATE = 2  # Rated power of PV (kW)
N_PV = 0.9  # Efficiency factor for PV
PHI_PV = 0.24  # Loss factor for PV

# Energy storage system (ESS) parameters
P_ESS_CH_MAX = 2  # Maximum charging power (kW)
P_ESS_DCH_MAX = 2  # Maximum discharging power (kW)
ESS_DOD = 0.8  # Depth of discharge
SOC_ESS_MAX = 5  # Maximum state of charge (kWh)
N_ESS_CH = 0.98  # Charging efficiency
N_ESS_DCH = 0.98  # Discharging efficiency
SOC_ESS_MIN = (1 - ESS_DOD) * SOC_ESS_MAX  # Minimum state of charge considering DoD
SOC_ESS_SETPOINT = SOC_ESS_MAX  # Reference state of charge for ESS
PHI_ESS = 1e-6  # Loss factor for ESS
SOC_ESS_THRESHOLD = SOC_ESS_SETPOINT - N_ESS_CH * P_ESS_CH_MAX
PENALTY_COEFFICIENT = 100  # Factor to penalize bad actions

# Electric vehicle (EV) parameters
P_EV_CH_MAX = 3.3  # Maximum charging power (kW)
P_EV_DCH_MAX = 3.3  # Maximum discharging power (kW)
EV_DOD = 0.8  # Depth of discharge
SOC_EV_MAX = 24  # Maximum state of charge (kWh)
N_EV_CH = 0.98  # Charging efficiency
N_EV_DCH = 0.98  # Discharging efficiency
SOC_EV_MIN = (1 - EV_DOD) * SOC_EV_MAX  # Minimum state of charge considering DoD
SOC_EV_SETPOINT = SOC_EV_MAX  # Reference state of charge for EV
PHI_EV = 1e-6
SOC_EV_THRESHOLD = SOC_EV_SETPOINT - N_EV_CH * P_EV_CH_MAX

def create_directories():
    """
    Ensure that all necessary directories exist. If not, create them.
    """
    os.makedirs(RAW_DATA_DIR, exist_ok=True)
    os.makedirs(GENERATED_DATA_DIR, exist_ok=True)

def print_config():
    """
    Utility function to print the current configuration settings.
    Useful for debugging and verification.
    """
    print("Microgrid Configuration Settings:")
    print(f"Time horizon: {T_NUM} hours")
    print(f"Time step: {DELTA_T} hours")
    print(f"Max SOC for ESS: {SOC_ESS_MAX} kWh")
    print(f"Min SOC for ESS: {SOC_ESS_MIN} kWh")
    # Add more configuration details as needed

# Automatically create directories when the module is imported
create_directories()

if __name__ == "__main__":
    print_config()