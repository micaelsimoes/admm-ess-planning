import os
from network import Network
from network_parameters import NetworkParameters


# ======================================================================================================================
#   Class NETWORK DATA -- Contains information of the Network over the planning period (years, days)
# ======================================================================================================================
class NetworkData:

    def __init__(self):
        self.name = str()
        self.data_dir = str()
        self.results_dir = str()
        self.plots_dir = str()
        self.diagrams_dir = str()
        self.years = dict()
        self.days = dict()
        self.num_instants = int()
        self.discount_factor = float()
        self.candidate_nodes = list()
        self.network = dict()
        self.params_file = str()
        self.params = NetworkParameters()
        self.cost_energy_p = dict()
        self.prob_market_scenarios = dict()

    def run_operational_planning_problem(self, models, from_warm_start=False):
        print(f'[INFO] Running SMOPF, Network {self.name}...')
        results = dict()
        for year in self.years:
            results[year] = dict()
            for day in self.days:
                print(f'[INFO] \t - Year {year}, Day {day}...')
                results[year][day] = self.network[year][day].run_smopf(models[year][day], self.params, from_warm_start=from_warm_start)
        return results

    def build_model(self):
        network_models = dict()
        for year in self.years:
            network_models[year] = dict()
            for day in self.days:
                network_models[year][day] = self.network[year][day].build_model(self.candidate_nodes, self.params)
        return network_models

    def read_network_data(self):
        _read_network_data(self)

    def read_network_parameters(self):
        filename = os.path.join(self.data_dir, self.name, self.params_file)
        self.params.read_parameters_from_file(filename)

    def process_results(self, model, results=dict()):
        return _process_results(self, model, results)


# ======================================================================================================================
#  NETWORK DATA read function
# ======================================================================================================================
def _read_network_data(network_planning):

    for year in network_planning.years:

        network_planning.network[year] = dict()

        for day in network_planning.days:

            # Create Network object
            network_planning.network[year][day] = Network()
            network_planning.network[year][day].name = network_planning.name
            network_planning.network[year][day].data_dir = network_planning.data_dir
            network_planning.network[year][day].results_dir = network_planning.results_dir
            network_planning.network[year][day].plots_dir = network_planning.plots_dir
            network_planning.network[year][day].diagrams_dir = network_planning.diagrams_dir
            network_planning.network[year][day].year = int(year)
            network_planning.network[year][day].day = day
            network_planning.network[year][day].num_instants = network_planning.num_instants
            network_planning.network[year][day].prob_market_scenarios = network_planning.prob_market_scenarios
            network_planning.network[year][day].cost_energy_p = network_planning.cost_energy_p[year][day]
            network_planning.network[year][day].operational_data_file = f'{network_planning.name}_{year}.xlsx'

            # Read info from file(s)
            network_planning.network[year][day].read_network_from_json_file()
            network_planning.network[year][day].read_network_operational_data_from_file()

            if network_planning.params.print_to_screen:
                network_planning.network[year][day].print_network_to_screen()
            if network_planning.params.plot_diagram:
                network_planning.network[year][day].plot_diagram()


# ======================================================================================================================
#  NETWORK PLANNING results functions
# ======================================================================================================================
def _process_results(network_planning, models, optimization_results):
    processed_results = dict()
    processed_results['results'] = dict()
    processed_results['of_value'] = _get_objective_function_value(network_planning, models)
    for year in network_planning.years:
        processed_results['results'][year] = dict()
        for day in network_planning.days:
            model = models[year][day]
            result = optimization_results[year][day]
            network = network_planning.network[year][day]
            processed_results['results'][year][day] = network.process_results(model, network_planning.params, network_planning.candidate_nodes, result)
    return processed_results


def _get_objective_function_value(network_planning, models):

    years = [year for year in network_planning.years]

    of_value = 0.0
    initial_year = years[0]
    for y in range(len(network_planning.years)):
        year = years[y]
        num_years = network_planning.years[year]
        annualization = 1 / ((1 + network_planning.discount_factor) ** (int(year) - int(initial_year)))
        for day in network_planning.days:
            num_days = network_planning.days[day]
            network = network_planning.network[year][day]
            model = models[year][day]
            of_value += annualization * num_days * num_years * network.compute_objective_function_value(model, network_planning.params)
    return of_value
