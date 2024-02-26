from helper_functions import *


# ======================================================================================================================
#  Class Planning Parameters
# ======================================================================================================================
class PlanningParameters:

    def __init__(self):
        self.tol_abs = 1e3
        self.tol_rel = 1e-2
        self.num_max_iters = 1000
        self.rho_s = 1.00
        self.rho_e = 1.00

    def read_parameters_from_file(self, filename):
        params_data = convert_json_to_dict(read_json_file(filename))
        self.tol_abs = float(params_data['tol_abs'])
        self.tol_rel = float(params_data['tol_rel'])
        self.num_max_iters = int(params_data['num_max_iters'])
