import numpy as np
import gurobipy as gp
from gurobipy import GRB
from typing import Dict, Optional
from .. import config as cfg

from .util import extract_results
from .components.utility_grid import Grid
from .components.renewables import PV
from .components.energy_storage import ESS
from .components.electric_vehicle import EV

class HomeEnergyManagementSystem:
    def __init__(self):
        # Time Horizon
        self.T_num = cfg.T_NUM
        self.T_set = cfg.T_SET
        self.delta_t = cfg.DELTA_T

        # Initialize the components of the microgrid
        self.grid = Grid(cfg.P_GRID_PUR_MAX, cfg.P_GRID_EXP_MAX, cfg.R_GRID_PUR, cfg.R_GRID_EXP, cfg.PHI_RTP, cfg.T_NUM, cfg.T_SET, cfg.DELTA_T)
        self.pv = PV(cfg.P_PV_RATE, cfg.N_PV, cfg.PHI_PV, cfg.T_SET)
        self.ess = ESS(cfg.P_ESS_CH_MAX, cfg.P_ESS_DCH_MAX, cfg.SOC_ESS_MAX, cfg.SOC_ESS_MIN, cfg.SOC_ESS_SETPOINT, cfg.N_ESS_CH, cfg.N_ESS_DCH, cfg.PHI_ESS, cfg.T_NUM, cfg.T_SET, cfg.DELTA_T)
        self.ev = EV(cfg.P_EV_CH_MAX, cfg.P_EV_DCH_MAX, cfg.SOC_EV_MAX, cfg.SOC_EV_MIN, cfg.SOC_EV_SETPOINT, cfg.N_EV_CH, cfg.N_EV_DCH, cfg.T_NUM, cfg.T_SET, cfg.DELTA_T)

    def optim(self, rtp: np.ndarray, p_pv_max: np.ndarray, p_if: np.ndarray, initial_soc_ev_percentage: float, ArriveTime: int, DepartureTime: int) -> Dict[str, np.ndarray]:
        """Optimization method for the microgrid."""
        model = gp.Model()
        model.ModelSense = GRB.MINIMIZE
        model.Params.LogToConsole = 0

        # Initialize variables for each component
        p_grid_pur, p_grid_exp, u_grid_pur, u_grid_exp = self.grid.add_variables(model)
        p_pv = self.pv.add_variables(model, p_pv_max)
        p_ess_ch, p_ess_dch, p_ess, u_ess_ch, u_ess_dch, soc_ess, F_ess = self.ess.add_variables(model)
        p_ev_ch, p_ev_dch, u_ev_ch, u_ev_dch, soc_ev = self.ev.add_variables(model)

        # Define PEV range operation
        initial_soc_ev, t_ev_arrive, t_ev_depart, ev_time_range = self.ev.get_ev_availablity(initial_soc_ev_percentage, ArriveTime, DepartureTime)

        # Add constraints for each component
        self.grid.add_constraints(model, p_grid_pur, p_grid_exp, u_grid_pur, u_grid_exp)
        self.ess.add_constraints(model, p_ess_ch, p_ess_dch, p_ess, u_ess_ch, u_ess_dch, soc_ess, F_ess)
        self.ev.add_constraints(model, p_ev_ch, p_ev_dch, u_ev_ch, u_ev_dch, soc_ev, initial_soc_ev, t_ev_arrive, t_ev_depart, ev_time_range)

        # Energy balance
        for t in self.T_set:
            model.addConstr(p_grid_pur[t] + p_pv[t] + p_ess_dch[t] + p_ev_dch[t] == p_grid_exp[t] + p_if[t] + p_ess_ch[t] + p_ev_ch[t])

        # Cost exchange with utility grid
        F_grid = self.grid.get_cost(rtp, p_grid_pur, p_grid_exp)

        # Define problem and solve
        model.setObjective(F_grid)
        model.optimize()

        if model.status == GRB.OPTIMAL:
            results = {
                'ObjVal': model.ObjVal,
                'p_grid_pur': extract_results(p_grid_pur, self.T_set),
                'p_grid_exp': extract_results(p_grid_exp, self.T_set),
                'u_grid_pur': extract_results(u_grid_pur, self.T_set),
                'u_grid_exp': extract_results(u_grid_exp, self.T_set),
                'p_pv': extract_results(p_pv, self.T_set),
                'p_ess_ch': extract_results(p_ess_ch, self.T_set),
                'p_ess_dch': extract_results(p_ess_dch, self.T_set),
                'u_ess_ch': extract_results(u_ess_ch, self.T_set),
                'u_ess_dch': extract_results(u_ess_dch, self.T_set),
                'soc_ess': extract_results(soc_ess, self.T_set),
                'p_ev_ch': extract_results(p_ev_ch, self.T_set),
                'p_ev_dch': extract_results(p_ev_dch, self.T_set),
                'u_ev_ch': extract_results(u_ev_ch, self.T_set),
                'u_ev_dch': extract_results(u_ev_dch, self.T_set),
                'soc_ev': extract_results(soc_ev, self.T_set),
                'ev_time_range': ev_time_range,
                'rtp': rtp,
                'p_pv_max': p_pv_max,
                'p_if': p_if,
            }

        else:
            raise RuntimeError(f"Optimization was unsuccessful. Model status: {model.status}")

        return results