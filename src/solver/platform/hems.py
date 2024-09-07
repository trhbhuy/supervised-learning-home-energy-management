import numpy as np
import gurobipy as gp
from gurobipy import GRB
from typing import Dict, Optional
from .. import config as cfg

from .components.utility_grid import Grid
from .components.renewables import PV
from .components.energy_storage import ESS
from .components.electric_vehicle import EV

from .util import extract_results, create_results_dict

class HomeEnergyManagementSystem:
    def __init__(self):
        """
        Initializes the Home Energy Management System (HEMS) with various parameters.
        """
        # Time settings
        self.T_num = cfg.T_NUM
        self.T_set = cfg.T_SET
        self.delta_t = cfg.DELTA_T

        # Initialize the components of the isolated microgrid
        self.grid = Grid(cfg.T_NUM, cfg.T_SET, cfg.DELTA_T, cfg.P_GRID_PUR_MAX, cfg.P_GRID_EXP_MAX, cfg.PHI_RTP)
        self.pv = PV(cfg.T_NUM, cfg.T_SET, cfg.DELTA_T, cfg.P_PV_RATE, cfg.N_PV)
        self.ess = ESS(cfg.T_NUM, cfg.T_SET, cfg.DELTA_T, cfg.P_ESS_CH_MAX, cfg.P_ESS_DCH_MAX, cfg.N_ESS_CH, cfg.N_ESS_DCH, cfg.SOC_ESS_MAX, cfg.SOC_ESS_MIN, cfg.SOC_ESS_SETPOINT, enable_cost_modeling=True, phi_ess=cfg.PHI_ESS)
        self.ev = EV(cfg.T_NUM, cfg.T_SET, cfg.DELTA_T, cfg.P_EV_CH_MAX, cfg.P_EV_DCH_MAX, cfg.N_EV_CH, cfg.N_EV_DCH, cfg.SOC_EV_MAX, cfg.SOC_EV_MIN, cfg.SOC_EV_SETPOINT)

    def optim(self, rtp: np.ndarray, p_pv_max: np.ndarray, p_if: np.ndarray, soc_ev_init_perc: float, ArriveTime: int, DepartureTime: int) -> Dict[str, np.ndarray]:
        """HEMS optimization model."""
        model = gp.Model()
        model.ModelSense = GRB.MINIMIZE
        model.Params.LogToConsole = 0

        # Define EV availability
        t_ev_arrive, t_ev_depart, ev_time_range, soc_ev_init = self.ev.get_ev_availablity(ArriveTime, DepartureTime, soc_ev_init_perc=soc_ev_init_perc)

        # Initialize variables for each component
        p_grid_pur, p_grid_exp, u_grid_pur, u_grid_exp = self.grid.add_variables(model)
        p_pv = self.pv.add_variables(model, p_pv_max)
        p_ess_ch, p_ess_dch, u_ess_ch, u_ess_dch, soc_ess, p_ess, F_ess = self.ess.add_variables(model)
        p_ev_ch, p_ev_dch, u_ev_ch, u_ev_dch, soc_ev = self.ev.add_variables(model)            

        # Add constraints for each component
        self.grid.add_constraints(model, p_grid_pur, p_grid_exp, u_grid_pur, u_grid_exp)

        self.ess.add_constraints(model, p_ess_ch, p_ess_dch, u_ess_ch, u_ess_dch, soc_ess, p_ess, F_ess)
        self.ev.add_constraints(model, p_ev_ch, p_ev_dch, u_ev_ch, u_ev_dch, soc_ev, t_ev_arrive, t_ev_depart, ev_time_range, soc_ev_init)

        # Energy balance
        for t in self.T_set:
            model.addConstr(p_grid_pur[t] + p_pv[t] + p_ess_dch[t] + p_ev_dch[t] == p_grid_exp[t] + p_if[t] + p_ess_ch[t] + p_ev_ch[t])

        # Cost exchange with utility grid
        F_grid = self.grid.get_cost(rtp, p_grid_pur, p_grid_exp)

        # Define problem and solve
        model.setObjective(F_grid)
        model.optimize()

        if model.status == GRB.OPTIMAL:
            variable_dict = {
                'p_grid_pur': p_grid_pur,
                'p_grid_exp': p_grid_exp,
                'u_grid_pur': u_grid_pur,
                'u_grid_exp': u_grid_exp,
                'p_pv': p_pv,
                'p_ess_ch': p_ess_ch,
                'p_ess_dch': p_ess_dch,
                'u_ess_ch': u_ess_ch,
                'u_ess_dch': u_ess_dch,
                'soc_ess': soc_ess,
                'p_ev_ch': p_ev_ch,
                'p_ev_dch': p_ev_dch,
                'u_ev_ch': u_ev_ch,
                'u_ev_dch': u_ev_dch,
                'soc_ev': soc_ev
            }

            # Call the function to create the results dictionary
            results = create_results_dict(model, variable_dict, self.T_set)

            # Add other values like ev_time_range, rtp, etc. separately
            results.update({
                'ev_time_range': ev_time_range,
                'rtp': rtp,
                'p_pv_max': p_pv_max,
                'p_if': p_if,
            })

        else:
            raise RuntimeError(f"Optimization was unsuccessful. Model status: {model.status}")

        return results