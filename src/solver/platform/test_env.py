import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Callable, Dict, Tuple, Optional
import gurobipy as gp
from gurobipy import GRB
from .components.electric_vehicle import EV

from .. import config as cfg
from ..methods.data_loader import load_data
from .util import scaler_loader, check_boundary_constraint, check_setpoint

class SmartHomeEnv(gym.Env):
    def __init__(self):
        """Initialize the environment."""
        # Load and initialize the parameters
        self._init_params()
    
        # Load the simulation data
        self.data = load_data(is_train=False)
        self.num_scenarios = len(self.data['soc_ev_init'])
        print(f"Number of scenarios: {self.num_scenarios}")

        # Initialize EV
        self.ev = EV(cfg.T_NUM, cfg.T_SET, cfg.DELTA_T, cfg.P_EV_CH_MAX, cfg.P_EV_DCH_MAX, cfg.N_EV_CH, cfg.N_EV_DCH, cfg.SOC_EV_MAX, cfg.SOC_EV_MIN, cfg.SOC_EV_SETPOINT)

        # Load state and action scalers
        self.state_scaler, self.action_scaler = scaler_loader()

        # Define observation space (normalized to [0, 1])
        observation_dim = 4
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(observation_dim,), dtype=np.float32)

        # Define action space (normalized to [0, 1])
        action_dim = 1
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(action_dim,), dtype=np.float32)

    def reset(self, seed=0):
        """Reset the environment to an initial state."""
        self.scenario_seed = int(seed)
        self.time_step = 0

        # Calculate the index for the scenario data
        index = self._get_index(self.scenario_seed, self.time_step)

        # Define EV availability
        self._init_ev_availability()

        # Initialize state
        initial_state = np.array([
            self.time_step,
            self.data['rtp'][index],
            self.data['p_if'][index] - self.data['p_pv_max'][index],
            self.soc_ess_setpoint,
            0.0
        ], dtype=np.float32)

        # Normalize the initial state
        self.state = self.state_scaler.transform([initial_state])[0].astype(np.float32)

        return self.state, {}
    
    def step(self, action):
        """Take an action and return the next state, reward, and termination status."""
        # Inverse transform the state from normalized form
        current_state = self.state_scaler.inverse_transform([self.state])[0].astype(np.float32)

        # Decompose the state into individual variables
        time_step, rtp, p_net, soc_ess_tempt, soc_ev_tempt = current_state
        time_step = int(np.round(time_step))

        # Fetch the data for the current time step
        base_idx = self._get_index(self.scenario_seed, time_step)

        # Inverse transform the action using the scaler
        action_pred = self.action_scaler.inverse_transform(action.reshape(1, -1))[0]

        # Clip actions to be within their respective limits
        ess_action, ev_action = np.clip(action_pred[:2],
                                        [-self.p_ess_dch_max, -self.p_ev_ch_max],
                                        [self.p_ev_ch_max, self.p_ev_ch_max]) 

        p_ess_ch, p_ess_dch, soc_ess = self._update_ess(time_step, ess_action, soc_ess_tempt)
        p_ev_ch, p_ev_dch, soc_ev = self._update_ev(time_step, ev_action, soc_ev_tempt)

        # Solve the MILP optimization problem
        p_grid_pur, p_grid_exp, u_grid_pur, u_grid_exp, p_pv, reward = self._optim(rtp, 
                                                                                  self.data['p_pv_max'][base_idx], 
                                                                                  self.data['p_if'][base_idx], 
                                                                                  p_ess_ch, p_ess_dch, 
                                                                                  p_ev_ch, p_ev_dch)

        # Calculate the reward and penalties
        cumulative_penalty = self._get_penalty(time_step, p_grid_pur, p_grid_exp, soc_ess, soc_ev, self.ev_time_range[time_step])
        reward += cumulative_penalty * self.penalty_coefficient

        # Prepare next state
        next_state, terminated = self._get_obs(time_step, soc_ess, soc_ev)

        # Update the state for the next step
        self.state = self.state_scaler.transform([next_state])[0].astype(np.float32)

        return self.state, reward, terminated, False, {
            "p_grid_pur": p_grid_pur,
            "p_grid_exp": p_grid_exp,
            "u_grid_pur": u_grid_pur,
            "u_grid_exp": u_grid_exp,
            "p_pv": p_pv,
            "p_if": self.data['p_if'][base_idx],
            "p_ess_ch": p_ess_ch,
            "p_ess_dch": p_ess_dch,
            "soc_ess": soc_ess,
            "p_ev_ch": p_ev_ch,
            "p_ev_dch": p_ev_dch,
            "soc_ev": soc_ev,
        }

    def _get_obs(self, time_step, soc_ess, soc_ev):
        """Prepare the next state and determine if the episode has terminated."""
        # Increment the time step.
        time_step += 1

        # Determine if the episode has terminated.
        terminated = time_step >= self.T_num

        # Update EV state of charge based on arrival and departure times.
        if time_step == self.t_ev_arrive:
            soc_ev = self.soc_ev_init
        elif time_step > self.t_ev_depart:
            soc_ev = 0

        # Prepare the next state if the episode is ongoing.
        if not terminated:
            base_idx = self.scenario_seed * self.T_num + time_step
            next_state = np.array([
                time_step,
                self.data['rtp'][base_idx],
                self.data['p_if'][base_idx] - self.data['p_pv_max'][base_idx],
                soc_ess,
                soc_ev
            ], dtype=np.float32)
        else:
            next_state = np.array([time_step, 0, 0, 0, 0], dtype=np.float32)

        return next_state, terminated

    def _update_ess(self, time_step, action, soc_ess_tempt):
        """Update the ESS state based on the action taken."""
        # Initialize charge and discharge power
        p_ess_ch, p_ess_dch = 0.0, 0.0
        
        if time_step == 0:
            # No ESS action allowed at the first timestep
            pass
        elif time_step == self.T_set[-1]:
            # Limit the charge power to not exceed the SOC setpoint
            p_ess_ch = min((self.soc_ess_setpoint - soc_ess_tempt) / (self.n_ess_ch * self.delta_t), self.p_ess_ch_max)
        else:
            # Charge or discharge ESS based on action
            p_ess_ch = max(action, 0)
            p_ess_dch = -min(action, 0)

            # Calculate potential SOC after applying action
            soc_ess = soc_ess_tempt + self.delta_t * (p_ess_ch * self.n_ess_ch - p_ess_dch / self.n_ess_dch)

            # Postprocess action to meet all ESS bound constraints
            p_ess_ch, p_ess_dch = self._postprocess_bound(p_ess_ch, p_ess_dch, soc_ess, soc_ess_tempt, self.p_ess_ch_max, self.p_ess_dch_max, self.n_ess_ch, self.n_ess_dch, self.soc_ess_max, self.soc_ess_min)

            # # Special condition for the second-to-last timestep
            # if time_step == self.T_num - 2:
            #     p_ess_ch, p_ess_dch = self._postprocess_ess_setpoint(p_ess_ch, p_ess_dch, soc_ess, soc_ess_tempt)

        # Update SOC based on adjusted powers
        soc_ess = soc_ess_tempt + self.delta_t * (p_ess_ch * self.n_ess_ch - p_ess_dch / self.n_ess_dch)

        return p_ess_ch, p_ess_dch, soc_ess

    def _update_ev(self, time_step, action, soc_ev_tempt):
        """Update the EV state based on the action taken."""
        # Initialize charge and discharge power
        p_ev_ch, p_ev_dch = 0.0, 0.0

        if self.ev_time_range[time_step] == 0:
            pass
        else:
            # Charge or discharge ev based on action
            p_ev_ch = max(action, 0)
            p_ev_dch = -min(action, 0)

            # Calculate potential SOC after applying action
            soc_ev = soc_ev_tempt + self.delta_t * (p_ev_ch * self.n_ev_ch - p_ev_dch / self.n_ev_dch)

            # Postprocess action to meet all EV bound constraints
            p_ev_ch, p_ev_dch = self._postprocess_bound(p_ev_ch, p_ev_dch, soc_ev, soc_ev_tempt, self.p_ev_ch_max, self.p_ev_dch_max, self.n_ev_ch, self.n_ev_dch, self.soc_ev_max, self.soc_ev_min)

        # Update SOC based on adjusted powers
        soc_ev = soc_ev_tempt + self.delta_t * (p_ev_ch * self.n_ev_ch - p_ev_dch / self.n_ev_dch)

        return p_ev_ch, p_ev_dch, soc_ev

    def _postprocess_bound(self, p_ch, p_dch, soc, soc_tempt, p_ch_max, p_dch_max, n_ch, n_dch, soc_max, soc_min):
        """Adjust charging and discharging powers based on SOC constraints."""

        # SOC is above the maximum limit.
        if soc > soc_max:
            p_ch = min((soc_max - soc_tempt) / (n_ch * self.delta_t), p_ch_max)
            p_dch = 0

        # SOC is below the minimum limit.
        elif soc < soc_min:
            p_ch = 0
            p_dch = min((soc_tempt - soc_min) * n_dch / self.delta_t, p_dch_max)

        else:
            p_ch = min(p_ch, p_ch_max)
            p_dch = min(p_dch, p_dch_max)

        return p_ch, p_dch

    def _postprocess_ess_setpoint(self, p_ess_ch, p_ess_dch, soc_ess, soc_ess_tempt):
        """Handle special conditions for ESS at the second-to-last timestep."""
        if soc_ess < self.soc_ess_threshold:
            if soc_ess_tempt < self.soc_ess_threshold:
                p_ess_ch = min((self.soc_ess_threshold - soc_ess_tempt) / (self.n_ess_ch * self.delta_t), self.p_ess_ch_max)
                p_ess_dch = 0
            elif soc_ess_tempt > self.soc_ess_threshold:
                p_ess_ch = 0
                p_ess_dch = min((soc_ess_tempt - self.soc_ess_threshold) * self.n_ess_dch / self.delta_t, self.p_ess_dch_max)

        return p_ess_ch, p_ess_dch

    def _get_penalty(self, time_step, p_grid_pur, p_grid_exp, soc_ess, soc_ev, ev_time_range):
        """Calculate penalties for boundary and ramp rate violations for the generator and ESS."""
        # Grid penalties
        grid_penalty = 0
        grid_penalty += check_boundary_constraint(p_grid_pur, 0, self.p_grid_pur_max)
        grid_penalty += check_boundary_constraint(p_grid_exp, 0, self.p_grid_exp_max)

        # ESS penalties
        ess_penalty = 0
        ess_penalty += check_boundary_constraint(soc_ess, self.soc_ess_min, self.soc_ess_max)

        if time_step == 0 or time_step == self.T_set[-1]:
            ess_penalty += check_setpoint(soc_ess, self.soc_ess_setpoint)

        # ESS penalties
        ev_penalty = 0
        ev_penalty += check_boundary_constraint(soc_ev, self.soc_ev_min * ev_time_range, self.soc_ev_max)

        # if time_step == self.t_ev_depart:
        #     ev_penalty += check_setpoint(soc_ev, self.soc_ev_setpoint)

        return grid_penalty + ess_penalty + ev_penalty

    def _optim(self, rtp, p_pv_max, p_if, p_ess_ch, p_ess_dch, p_ev_ch, p_ev_dch):
        """Optimization method for the microgrid."""
        # Create a new model
        model = gp.Model()
        model.ModelSense = GRB.MINIMIZE
        model.Params.LogToConsole = 0

        ## Utility grid modeling
        p_grid_pur = model.addVar(vtype=GRB.CONTINUOUS, name="p_grid_pur")
        u_grid_pur = model.addVar(vtype=GRB.BINARY, name="u_grid_pur")
        p_grid_exp = model.addVar(vtype=GRB.CONTINUOUS, name="p_grid_exp")
        u_grid_exp = model.addVar(vtype=GRB.BINARY, name="u_grid_exp")

        # Grid constraints
        model.addConstr(p_grid_pur <= self.p_grid_pur_max * u_grid_pur)
        model.addConstr(p_grid_exp <= self.p_grid_exp_max * u_grid_exp)
        model.addConstr(u_grid_pur + u_grid_exp >= 0)
        model.addConstr(u_grid_pur + u_grid_exp <= 1)

        ## Solar PV scheduled
        p_pv = model.addVar(lb=0, ub=p_pv_max, vtype=GRB.CONTINUOUS, name="p_pv")

        # Energy balance
        model.addConstr(p_grid_pur + p_pv + p_ess_dch + p_ev_dch == p_grid_exp + p_if + p_ess_ch + p_ev_ch)

        # Cost exchange with utility grid
        F_grid = self.delta_t * (p_grid_pur * rtp - p_grid_exp * rtp * self.phi_rtp)

        # Define problem and solve
        model.setObjective(F_grid)
        model.optimize()

        # return results
        return p_grid_pur.x, p_grid_exp.x, u_grid_pur.x, u_grid_exp.x, p_pv.x, model.objVal

    def _init_params(self):
        """Initialize constants and parameters from the scenario configuration."""
        # Time settings
        self.T_num = cfg.T_NUM
        self.T_set = cfg.T_SET
        self.delta_t = cfg.DELTA_T
        self.penalty_coefficient = cfg.PENALTY_COEFFICIENT

        # Parameters for Grid Exchange
        self.p_grid_pur_max = cfg.P_GRID_PUR_MAX
        self.p_grid_exp_max = cfg.P_GRID_EXP_MAX
        self.phi_rtp = cfg.PHI_RTP

        # Energy Storage System (ESS) Parameters
        self.p_ess_ch_max = cfg.P_ESS_CH_MAX
        self.p_ess_dch_max = cfg.P_ESS_DCH_MAX
        self.soc_ess_max = cfg.SOC_ESS_MAX
        self.n_ess_ch = cfg.N_ESS_CH
        self.n_ess_dch = cfg.N_ESS_DCH
        self.soc_ess_min = cfg.SOC_ESS_MIN
        self.soc_ess_setpoint = cfg.SOC_ESS_SETPOINT
        self.soc_ess_threshold = cfg.SOC_ESS_THRESHOLD

        # Electric Vehicle (EV) Parameters
        self.p_ev_ch_max = cfg.P_EV_CH_MAX
        self.p_ev_dch_max = cfg.P_EV_DCH_MAX
        self.soc_ev_max = cfg.SOC_EV_MAX
        self.n_ev_ch = cfg.N_EV_CH
        self.n_ev_dch = cfg.N_EV_DCH
        self.soc_ev_min = cfg.SOC_EV_MIN
        self.soc_ev_setpoint = cfg.SOC_EV_SETPOINT
        self.soc_ev_threshold = cfg.SOC_EV_THRESHOLD

    def _get_index(self, scenario: int, time_step: int) -> int:
        """Get index for scenario data based on time step and scenario seed."""
        return scenario * self.T_num + time_step

    def _init_ev_availability(self):
        """Initialize electric vehicle availability for the current scenario."""
        self.t_ev_arrive, self.t_ev_depart, self.ev_time_range, self.soc_ev_init = self.ev.get_ev_availablity(
            self.data['t_ev_arrive'][self.scenario_seed], 
            self.data['t_ev_depart'][self.scenario_seed], 
            soc_ev_init_perc=self.data['soc_ev_init'][self.scenario_seed]
        )