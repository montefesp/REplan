from numpy import arange
from pyomo.environ import ConcreteModel, Var, Constraint, Objective, minimize, maximize, NonNegativeReals
from pyomo.opt import ProblemFormat, SolverFactory
from os.path import join, dirname, abspath
from shapely.ops import cascaded_union
from shapely.geometry import MultiPoint
import pandas as pd
from src.resite.utils import custom_log, init_folder
from src.resite.tools import filter_points, compute_capacity_factors, read_database, \
    get_legacy_capacity, get_capacity_potential
from src.data.load.manager import retrieve_load_data
from src.data.geographics.manager import return_region_shape, return_points_in_shape, get_subregions, \
    display_polygons
from src.resite.tools import get_tech_points_tuples
from typing import List, Dict, Any
from shutil import copy
import pickle


class Model:

    def __init__(self, params):

        self.output_folder = init_folder(params['keep_files'])
        copy('config_model.yml', self.output_folder)
        copy('config_techs.yml', self.output_folder)

        self.technologies = params['technologies']
        self.regions = params['regions']
        self.timestamps = pd.date_range(params['time_slice'][0], params['time_slice'][1], freq='1H').values
        self.spatial_res = params['spatial_resolution']

        self.instance = None

    def build_input_data(self, filtering_layers: Dict[str, bool]):
        """Data pre-processing.

        Parameters:
        -----------
        filtering_layers: Dict[str, bool]
            Dictionary indicating if a given filtering layers needs to be applied. If the layer name is present as key and
            associated to a True boolean, then the corresponding is applied.
        """

        custom_log("Loading load")
        self.load_df = retrieve_load_data(self.regions, self.timestamps)

        custom_log("Getting region shapes")
        region_shapes = pd.DataFrame(index=self.regions, columns=['full'])
        all_subregions = []
        for region in self.regions:
            subregions = get_subregions(region)
            all_subregions += subregions
            shapes = return_region_shape(region, subregions)
            region_shapes.loc[region, 'full'] = cascaded_union([shapes['onshore'], shapes['offshore']])
        regions_shapes_union = cascaded_union(region_shapes['full'].values)

        # TODO: Need to remove the first init_points by downloading new data
        path_resource_data = join(dirname(abspath(__file__)), '../../data/resource/' + str(self.spatial_res))
        database = read_database(path_resource_data)
        init_points = list(zip(database.longitude.values, database.latitude.values))
        init_points = return_points_in_shape(regions_shapes_union, self.spatial_res, init_points)

        custom_log("Filtering coordinates")
        self.tech_points_dict = filter_points(self.technologies, init_points, self.spatial_res, filtering_layers)

        custom_log("Get existing legacy capacity")
        tech_with_legacy_data = list(set(self.technologies).intersection(['wind_onshore', 'wind_offshore', 'pv_utility']))
        existing_capacity_dict = get_legacy_capacity(tech_with_legacy_data, all_subregions, init_points, self.spatial_res)

        # Update filtered points
        for tech in existing_capacity_dict:
            if existing_capacity_dict[tech] is not None:
                self.tech_points_dict[tech] += list(existing_capacity_dict[tech].keys())
            # Remove duplicates
            self.tech_points_dict[tech] = list(set(self.tech_points_dict[tech]))

        # Remove techs that have no points associated to them
        self.tech_points_dict = {k: v for k, v in self.tech_points_dict.items() if len(v) > 0}

        # Associating coordinates to regions
        # regions_coords_dict = {region: set() for region in regions}
        self.region_tech_points_dict = {i: set() for i, region in enumerate(self.regions)}
        for tech, coords in self.tech_points_dict.items():
            coords_multipoint = MultiPoint(coords)
            for i, region in enumerate(self.regions):
                coords_in_region = coords_multipoint.intersection(region_shapes.loc[region, 'full'])
                coords_in_region = [(tech, (point.x, point.y)) for point in coords_in_region] \
                    if isinstance(coords_in_region, MultiPoint) \
                    else [(tech, (coords_in_region.x, coords_in_region.y))]
                self.region_tech_points_dict[i] = self.region_tech_points_dict[i].union(set(coords_in_region))

        # Create dataframe with existing capacity
        self.tech_points_tuples = get_tech_points_tuples(self.tech_points_dict)
        existing_capacity_ds = pd.Series(0., index=pd.MultiIndex.from_tuples(self.tech_points_tuples))
        for tech, coord in existing_capacity_ds.index:
            if tech in existing_capacity_dict and existing_capacity_dict[tech] is not None \
                    and coord in existing_capacity_dict[tech]:
                existing_capacity_ds[tech, coord] = existing_capacity_dict[tech][coord]

        custom_log("Compute cap factor")
        self.cap_factor_df = compute_capacity_factors(self.tech_points_dict, self.spatial_res, self.timestamps)

        custom_log("Compute capacity potential per node")
        self.cap_potential_ds = get_capacity_potential(self.tech_points_dict, self.spatial_res, self.regions,
                                                       existing_capacity_ds)

        # Compute percentage of existing capacity and set to 1. when capacity is zero
        existing_cap_percentage_ds = existing_capacity_ds.divide(self.cap_potential_ds)
        self.existing_cap_percentage_ds = existing_cap_percentage_ds.fillna(1.)


    # TODO:
    #  - create three functions, so that the docstring at the beginning of each function explain the model
    #  -> modeling
    def build_model(self, formulation: str, deployment_vector: List[float], write_lp: bool = False):
        """Model build-up.

        Parameters:
        ------------
        formulation: str
            Formulation of the optimization problem to solve
        deployment_vector: List[float]
            # TODO: this is dependent on the formulation so maybe we should create a different function for each formulation
        output_folder: str
            Path towards output folder
        write_lp : bool (default: False)
            If True, the model is written to an .lp file.
        """

        load = self.load_df.values
        # Maximum generation that can be produced if max capacity installed
        generation_potential_df = self.cap_factor_df*self.cap_potential_ds
        # generation_potential = generation_potential_df.values
        tech_points_tuples = [(tech, coord[0], coord[1]) for tech, coord in self.tech_points_tuples]

        model = ConcreteModel()

        if formulation == 'meet_RES_targets_year_round':  # TODO: probaly shouldn't be called year round

            # Variables for the portion of demand that is met at each time-stamp for each region
            model.x = Var(self.regions, arange(len(self.timestamps)), within=NonNegativeReals, bounds=(0, 1))
            # Variables for the portion of capacity at each location for each technology
            model.y = Var(tech_points_tuples, within=NonNegativeReals, bounds=(0, 1))

            # Create generation dictionary for building speed up
            region_generation_y_dict = dict.fromkeys(self.regions)
            for i, region in enumerate(self.regions):
                # Get generation potential for points in region for each techno
                tech_points_generation_potential = generation_potential_df[self.region_tech_points_dict[i]]
                region_ys = pd.Series([model.y[tech, loc] for tech, loc in self.region_tech_points_dict[i]],
                                      index=pd.MultiIndex.from_tuples(self.region_tech_points_dict[i]))
                region_generation = tech_points_generation_potential*region_ys
                region_generation_y_dict[region] = region_generation.sum(axis=1).values

            region_indexes = dict(zip(self.regions, arange(len(self.regions))))

            # Generation must be greater than x percent of the load in each region for each time step
            def generation_check_rule(model, region, t):
                return region_generation_y_dict[region][t] >= load[t, region_indexes[region]] * model.x[region, t]
            model.generation_check = Constraint(self.regions, arange(len(self.timestamps)), rule=generation_check_rule)

            # Percentage of capacity installed must be bigger than existing percentage
            def potential_constraint_rule(model, tech, lon, lat):
                return model.y[tech, lon, lat] >= self.existing_cap_percentage_ds[tech][(lon, lat)]
            model.potential_constraint = Constraint(tech_points_tuples, rule=potential_constraint_rule)

            # Impose a certain percentage of the load to be covered over the whole time slice
            covered_load_perc_per_region = dict(zip(self.regions, deployment_vector))

            def policy_target_rule(model, region):
                return sum(model.x[region, t] for t in arange(len(self.timestamps))) \
                       >= covered_load_perc_per_region[region] * len(self.timestamps)
            model.policy_target = Constraint(self.regions, rule=policy_target_rule)

            # Minimize the capacity that is deployed
            def objective_rule(model):
                return sum(model.y[tech, loc] * self.cap_potential_ds[tech, loc]
                           for tech, loc in self.cap_potential_ds.keys())
            model.objective = Objective(rule=objective_rule, sense=minimize)

        elif formulation == 'meet_RES_targets_hourly':

            # Variables for the portion of demand that is met at each time-stamp for each region
            model.x = Var(self.regions, arange(len(self.timestamps)), within=NonNegativeReals, bounds=(0, 1))
            # Variables for the portion of capacity at each location for each technology
            model.y = Var(tech_points_tuples, within=NonNegativeReals, bounds=(0, 1))

            # Create generation dictionary for building speed up
            region_generation_y_dict = dict.fromkeys(self.regions)
            for i, region in enumerate(self.regions):
                # Get generation potential for points in region for each techno
                tech_points_generation_potential = generation_potential_df[self.region_tech_points_dict[i]]
                region_ys = pd.Series([model.y[tech, loc] for tech, loc in self.region_tech_points_dict[i]],
                                index=pd.MultiIndex.from_tuples(self.region_tech_points_dict[i]))
                region_generation = tech_points_generation_potential*region_ys
                region_generation_y_dict[region] = region_generation.sum(axis=1).values

            region_indexes = dict(zip(self.regions, arange(len(self.regions))))

            # Generation must be greater than x percent of the load in each region for each time step
            def generation_check_rule(model, region, t):
                return region_generation_y_dict[region][t] >= load[t, region_indexes[region]] * model.x[region, t]
            model.generation_check = Constraint(self.regions, arange(len(self.timestamps)), rule=generation_check_rule)

            # Percentage of capacity installed must be bigger than existing percentage
            def potential_constraint_rule(model, tech, lon, lat):
                return model.y[tech, lon, lat] >= self.existing_cap_percentage_ds[tech][(lon, lat)]
            model.potential_constraint = Constraint(tech_points_tuples, rule=potential_constraint_rule)

            # Impose a certain percentage of the load to be covered for each time step
            covered_load_perc_per_region = dict(zip(self.regions, deployment_vector))

            # TODO: ask david, why are we multiplicating by len(timestamps)?
            def policy_target_rule(model, region, t):
                return model.x[region, t] >= covered_load_perc_per_region[region]   # * len(self.timestamps)
            model.policy_target = Constraint(self.regions, arange(len(self.timestamps)), rule=policy_target_rule)

            # Minimize the capacity that is deployed
            def objective_rule(model):
                return sum(model.y[tech, loc] * self.cap_potential_ds[tech, loc]
                           for tech, loc in self.cap_potential_ds.keys())
            model.objective = Objective(rule=objective_rule, sense=minimize)

        elif formulation == 'meet_demand_with_capacity':

            # Variables for the portion of demand that is met at each time-stamp for each region
            model.x = Var(self.regions, arange(len(self.timestamps)), within=NonNegativeReals, bounds=(0, 1))
            # Variables for the portion of capacity at each location for each technology
            model.y = Var(tech_points_tuples, within=NonNegativeReals, bounds=(0, 1))

            # Create generation dictionary for building speed up
            region_generation_y_dict = dict.fromkeys(self.regions)
            for i, region in enumerate(self.regions):
                # Get generation potential for points in region for each techno
                tech_points_generation_potential = generation_potential_df[self.region_tech_points_dict[i]]
                region_ys = pd.Series([model.y[tech, loc] for tech, loc in self.region_tech_points_dict[i]],
                                      index=pd.MultiIndex.from_tuples(self.region_tech_points_dict[i]))
                region_generation = tech_points_generation_potential * region_ys
                region_generation_y_dict[region] = region_generation.sum(axis=1).values

            region_indexes = dict(zip(self.regions, arange(len(self.regions))))

            # Generation must be greater than x percent of the load in each region for each time step
            def generation_check_rule(model, region, t):
                return region_generation_y_dict[region][t] >= load[t, region_indexes[region]] * model.x[region, t]
            model.generation_check = Constraint(self.regions, arange(len(self.timestamps)), rule=generation_check_rule)

            # Percentage of capacity installed must be bigger than existing percentage
            def potential_constraint_rule(model, tech, lon, lat):
                return model.y[tech, lon, lat] >= self.existing_cap_percentage_ds[tech][(lon, lat)]
            model.potential_constraint = Constraint(tech_points_tuples, rule=potential_constraint_rule)

            # Impose a certain installed capacity per technology
            required_installed_cap_per_tech = dict(zip(self.technologies, deployment_vector))

            def capacity_target_rule(model, tech: str):
                total_cap = sum(model.y[tech, loc] * self.cap_potential_ds[tech, loc]
                                for loc in self.tech_points_dict[tech])
                return total_cap >= required_installed_cap_per_tech[tech]
            model.capacity_target = Constraint(self.technologies, rule=capacity_target_rule)

            # Maximize the proportion of load that is satisfied
            def objective_rule(model):
                return sum(model.x[region, t] for region in self.regions
                           for t in arange(len(self.timestamps)))
            model.objective = Objective(rule=objective_rule, sense=maximize)

        else:
            raise ValueError(' This optimization setup is not available yet. Retry.')

        if write_lp:
            model.write(filename=join(self.output_folder, 'model.lp'),
                        format=ProblemFormat.cpxlp,
                        io_options={'symbolic_solver_labels': True})

        self.instance = model

    def solve_model(self, solver, solver_options):
        """
        Solve a model

        Parameters
        ----------
        solver: str
            Name of the solver to use
        solver_options: Dict[str, float]
            Dictionary of solver options name and value

        """
        opt = SolverFactory(solver)
        for key, value in solver_options.items():
            opt.options[key] = value
        opt.solve(self.instance, tee=True, keepfiles=False, report_timing=True,
                  logfile=join(self.output_folder, 'solver_log.log'))

    # TODO: shouldn't this function just return the points and not the capacity?
    def retrieve_selected_points(self, save_file):

        selected_tech_points_dict = {tech: {} for tech in self.technologies}

        tech_points_tuples = [(tech, coord[0], coord[1]) for tech, coord in self.tech_points_tuples]
        for tech, lon, lat in tech_points_tuples:
            y_value = self.instance.y[tech, lon, lat].value
            if y_value > 0.:
                cap = y_value * self.cap_potential_ds[tech, (lon, lat)]
                selected_tech_points_dict[tech][(lon, lat)] = cap

        # Remove tech for which no technology was selected
        selected_tech_points_dict = {k: v for k, v in selected_tech_points_dict.items() if len(v) > 0}

        if save_file:
            pickle.dump(selected_tech_points_dict, open(join(self.output_folder, 'output_model.p'), 'wb'))

        return selected_tech_points_dict

