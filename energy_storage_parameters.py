from solver_parameters import SolverParameters
from helper_functions import *


# ======================================================================================================================
#   Class SharedEnergyStorage
# ======================================================================================================================
class EnergyStorageParameters:
    def __init__(self):
        self.budget = 1e6                   # Budget, [m.u.]
        self.max_capacity = 0.00            # Maximum capacity (per ESS), [MVAh]
        self.min_se_factor = 0.10           # Minimum S/E factor (related to the ESS technology)
        self.max_se_factor = 4.00           # Maximum S/E factor (related to the ESS technology)
        self.eff_ch = 0.90                  # Charging efficiency, [0-1]
        self.eff_dch = 0.90                 # Discharging efficiency, [0-1]
        self.max_pf = 0.80                  # Maximum power factor
        self.min_pf = -0.80                 # Minimum power factor
        self.t_cal = 20                     # Calendar life of the ESS, [years]
        self.cl_nom = 3650                  # Cycle life, nominal, [number of cycles]
        self.dod_nom = 0.80                 # Depth-of-Discharge, nominal, [0-1]
        self.solver_params = SolverParameters()

    def read_parameters_from_file(self, filename):
        params_data = convert_json_to_dict(read_json_file(filename))
        self.budget = float(params_data['budget'])
        self.max_capacity = float(params_data['max_capacity'])
        self.min_se_factor = float(params_data['min_se_factor'])
        self.max_se_factor = float(params_data['max_se_factor'])
        self.eff_ch = float(params_data['eff_ch'])
        self.eff_dch = float(params_data['eff_dch'])
        self.max_pf = float(params_data['max_pf'])
        self.min_pf = float(params_data['min_pf'])
        self.t_cal = int(params_data['tcal'])
        self.cl_nom = int(params_data['cl_nom'])
        self.dod_nom = float(params_data['dod_nom'])
        self.solver_params.read_solver_parameters(params_data['solver'])
