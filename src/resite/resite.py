from os.path import join, dirname, abspath
from os import makedirs
from shapely.ops import cascaded_union
from shapely.geometry import MultiPoint
import pandas as pd
from src.data.legacy.manager import get_legacy_capacity
from src.data.resource.manager import read_resource_database, compute_capacity_factors
from src.data.land_data.manager import filter_points
from src.data.res_potential.manager import get_capacity_potential
from src.data.load.manager import get_load
from src.data.geographics.manager import return_region_shape, return_points_in_shape, get_subregions
from typing import List, Dict, Tuple, Any
from shutil import copy, rmtree
import yaml
from time import strftime
import pickle

import logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(asctime)s - %(message)s")
logger = logging.getLogger()


class Resite:

    def __init__(self, regions: List[str], technologies: List[str], tech_config: Dict[str, Any], timeslice: List[str],
                 spatial_resolution: float, keep_files: bool = False):
        """

        Parameters
        ----------
        regions: List[str]
            List of regions in which we want to site
        technologies: List[str]
            List of technologies for which we want to site
        tech_config: Dict[str, Any]
            Dictionary containing parameters configuration of each technology
        timeslice: List[str]
            List of 2 string containing starting and end date of the time horizon
        spatial_resolution: float
            Spatial resolution at which we want to site
        keep_files: bool (default: False)
            Whether to keep files at the end of the run
        """

        # self.params = params
        self.logger = logger
        self.keep_files = keep_files

        self.technologies = technologies
        self.regions = regions
        self.tech_config = tech_config
        self.timestamps = pd.date_range(timeslice[0], timeslice[1], freq='1H')
        self.spatial_res = spatial_resolution

        self.instance = None

    def init_output_folder(self, dir_name: str = None):
        """Initialize an output folder."""

        if dir_name is None:
            dir_name = join(dirname(abspath(__file__)), "../../output/resite/")
            self.output_folder = abspath(f"{dir_name}{strftime('%Y%m%d_%H%M%S')}/")
        else:
            self.output_folder = f"{dir_name}resite/"
        makedirs(self.output_folder)

        self.logger.info(f"Folder path is: {self.output_folder}")

        if not self.keep_files:
            self.logger.info('WARNING! Files will be deleted at the end of the run.')

    def __del__(self):
        """If self.keep_files is false, remove all outputs created during the run."""
        if not self.keep_files and hasattr(self, "output_folder"):
            rmtree(self.output_folder)

    # TODO: this is pretty messy - find a way to clean it up
    def build_input_data(self, use_ex_cap: bool, filtering_layers: Dict[str, bool]):
        """Data pre-processing.

        Parameters:
        -----------
        use_ex_cap: bool
            Whether to compute or not existing capacity and use it in optimization
        filtering_layers: Dict[str, bool]
            Dictionary indicating if a given filtering layers needs to be applied. If the layer name is present as key and
            associated to a True boolean, then the corresponding is applied.
        """

        self.use_ex_cap = use_ex_cap
        self.filtering_layers = filtering_layers

        # self.load_df = get_prepared_load(timestamps=self.timestamps, regions=self.regions)
        self.load_df = get_load(timestamps=self.timestamps, regions=self.regions, missing_data='interpolate')

        region_shapes = pd.DataFrame(index=self.regions, columns=['full'])
        all_subregions = []
        for region in self.regions:
            subregions = get_subregions(region)
            all_subregions += subregions
            shapes = return_region_shape(region, subregions)
            region_shapes.loc[region, 'full'] = cascaded_union([shapes['onshore'], shapes['offshore']])
        regions_shapes_union = cascaded_union(region_shapes['full'].values)

        # TODO: Need to remove the first init_points by downloading new data
        path_resource_data = join(dirname(abspath(__file__)),
                                  f"../../data/resource/source/era5-land/{self.spatial_res}")
        database = read_resource_database(path_resource_data)
        init_points = list(zip(database.longitude.values, database.latitude.values))
        init_points = return_points_in_shape(regions_shapes_union, self.spatial_res, init_points)

        self.tech_points_dict = filter_points(self.technologies, self.tech_config, init_points, self.spatial_res,
                                              filtering_layers)

        if use_ex_cap:
            tech_with_legacy_data = \
                list(set(self.technologies).intersection(['wind_onshore', 'wind_offshore', 'pv_utility', 'pv_residential']))
            existing_capacity_ds = get_legacy_capacity(tech_with_legacy_data, self.tech_config, all_subregions,
                                                       init_points, self.spatial_res)
            # Update filtered points
            for tech in tech_with_legacy_data:
                if tech in existing_capacity_ds.index.get_level_values(0):
                    self.tech_points_dict[tech] += list(existing_capacity_ds[tech].index)
                # Remove duplicates
                self.tech_points_dict[tech] = list(set(self.tech_points_dict[tech]))

        # Remove techs that have no points associated to them
        self.tech_points_dict = {k: v for k, v in self.tech_points_dict.items() if len(v) > 0}

        # Create dataframe with existing capacity
        self.tech_points_tuples = [(tech, point) for tech, points in self.tech_points_dict.items() for point in points]
        self.existing_capacity_ds = pd.Series(0., index=pd.MultiIndex.from_tuples(self.tech_points_tuples))
        if use_ex_cap:
            self.existing_capacity_ds.loc[existing_capacity_ds.index] = existing_capacity_ds.values

        self.cap_factor_df = compute_capacity_factors(self.tech_points_dict, self.tech_config,
                                                      self.spatial_res, self.timestamps)

        self.cap_potential_ds = get_capacity_potential(self.tech_points_dict, self.spatial_res, all_subregions,
                                                       self.existing_capacity_ds)

        # Compute percentage of existing capacity and set to 1. when capacity is zero
        existing_cap_percentage_ds = self.existing_capacity_ds.divide(self.cap_potential_ds)

        # Remove points which have zero potential capacity
        self.existing_cap_percentage_ds = existing_cap_percentage_ds.dropna()
        self.cap_potential_ds = self.cap_potential_ds[self.existing_cap_percentage_ds.index]
        self.cap_factor_df = self.cap_factor_df[self.existing_cap_percentage_ds.index]
        self.existing_capacity_ds = self.existing_capacity_ds[self.existing_cap_percentage_ds.index]
        self.tech_points_tuples = self.existing_cap_percentage_ds.index.values
        self.tech_points_dict = {}
        for tech, point in self.tech_points_tuples:
            if tech in self.tech_points_dict:
                self.tech_points_dict[tech] += [point]
            else:
                self.tech_points_dict[tech] = [point]

        # Maximum generation that can be produced if max capacity installed
        self.generation_potential_df = self.cap_factor_df * self.cap_potential_ds

        # Associating coordinates to regions
        self.region_tech_points_dict = {region: set() for region in self.regions}
        for tech, coords in self.tech_points_dict.items():
            coords_multipoint = MultiPoint(coords)
            for region in self.regions:
                coords_in_region = coords_multipoint.intersection(region_shapes.loc[region, 'full'])
                coords_in_region = [(tech, (point.x, point.y)) for point in coords_in_region] \
                    if isinstance(coords_in_region, MultiPoint) \
                    else [(tech, (coords_in_region.x, coords_in_region.y))]
                self.region_tech_points_dict[region] = self.region_tech_points_dict[region].union(set(coords_in_region))

    def build_model(self, modelling: str, formulation: str, deployment_vector: List[float], write_lp: bool = False):
        """Model build-up.

        Parameters:
        ------------
        modelling: str
            Choice of modelling language
        formulation: str
            Formulation of the optimization problem to solve
        deployment_vector: List[float]
            # TODO: this is dependent on the formulation so maybe we should create a different function for each formulation
        write_lp : bool (default: False)
            If True, the model is written to an .lp file.
        """

        if write_lp:
            self.init_output_folder()

        if formulation == 'meet_demand_with_capacity' and len(self.regions) != 1:
            raise ValueError('The selected formulation works for one region only!')
        elif formulation in ['meet_RES_targets_agg', 'meet_RES_targets_hourly', 'meet_RES_targets_daily',
                             'meet_RES_targets_weekly', 'meet_RES_targets_monthly', 'maximize_generation',
                             'maximize_aggr_cap_factor'] and len(deployment_vector) != len(self.regions):
            raise ValueError('For the selected formulation, the "regions" and "deployment_vector" '
                             'lists must have the same cardinality!')

        accepted_modelling = ['pyomo', 'docplex', 'gurobipy']
        assert modelling in accepted_modelling, f"Error: {modelling} is not available as modelling language. " \
                                                f"Accepted languages are {accepted_modelling}"

        self.modelling = modelling
        self.formulation = formulation
        self.deployment_vector = deployment_vector
        if self.modelling == 'pyomo':
            from src.resite.models.pyomo import build_model as build_pyomo_model
            build_pyomo_model(self, formulation, deployment_vector, write_lp)
        elif self.modelling == 'docplex':
            from src.resite.models.docplex import build_model as build_docplex_model
            build_docplex_model(self, formulation, deployment_vector, write_lp)
        elif self.modelling == 'gurobipy':
            from src.resite.models.gurobipy import build_model as build_gurobipy_model
            build_gurobipy_model(self, formulation, deployment_vector, write_lp)

    def solve_model(self):
        """
        # TODO: update comment
        Solve a model
        """
        if self.modelling == 'pyomo':
            from src.resite.models.pyomo import solve_model as solve_pyomo_model
            return solve_pyomo_model(self)
        elif self.modelling == 'docplex':
            from src.resite.models.docplex import solve_model as solve_docplex_model
            solve_docplex_model(self)
        elif self.modelling == 'gurobipy':
            from src.resite.models.gurobipy import solve_model as solve_gurobipy_model
            solve_gurobipy_model(self)

    def retrieve_solution(self) -> Dict[str, List[Tuple[float, float]]]:
        """
        Get points that were selected during the optimization

        Returns
        -------
        Dict[str, List[Tuple[float, float]]]
            Lists of selected points for each technology

        """
        if self.modelling == 'pyomo':
            from src.resite.models.pyomo import retrieve_solution as retrieve_pyomo_solution
            self.objective, self.selected_tech_points_dict, self.optimal_capacity_ds = retrieve_pyomo_solution(self)
        elif self.modelling == 'docplex':
            from src.resite.models.docplex import retrieve_solution as retrieve_docplex_solution
            self.objective, self.selected_tech_points_dict, self.optimal_capacity_ds = retrieve_docplex_solution(self)
        elif self.modelling == 'gurobipy':
            from src.resite.models.gurobipy import retrieve_solution as retrieve_gurobipy_solution
            self.objective, self.selected_tech_points_dict, self.optimal_capacity_ds = retrieve_gurobipy_solution(self)

        return self.selected_tech_points_dict

    def retrieve_sites_data(self):
        """
        This function returns the data for the optimal sites.

        Returns
        -------
        self.selected_existing_capacity_ds: pd.Series
            Pandas series giving for each (tech, coord) tuple in self.selected_tech_points_dict the existing
            capacity at these positions
        self.selected_capacity_potential_ds: pd.Series
            Pandas series giving for each (tech, coord) tuple in self.selected_tech_points_dict the capacity
            potential at these positions .
        self.selected_cap_factor_df: pd.DataFrame
            Pandas series indexed by time giving for each (tech, coord) tuple in self.selected_tech_points_dict
            its capacity factors time series

        """

        selected_tech_points_tuples = [(tech, point) for tech, points in self.selected_tech_points_dict.items()
                                       for point in points]

        self.selected_existing_capacity_ds = self.existing_capacity_ds.loc[selected_tech_points_tuples]
        self.selected_capacity_potential_ds = self.cap_potential_ds.loc[selected_tech_points_tuples]
        self.selected_cap_factor_df = self.cap_factor_df[selected_tech_points_tuples]

        return self.selected_existing_capacity_ds, self.selected_capacity_potential_ds, self.selected_cap_factor_df

    def save(self, params, dir_name: str = None):

        # TODO: shitty because model.lp or solver.log will be saved in a different file
        if not hasattr(self, 'output_folder'):
            self.init_output_folder(dir_name)

        # TODO: change this -> maybe we would need to have a function copying the parameters back to a file
        yaml.dump(params, open(f"{self.output_folder}config.yaml", 'w'))

        yaml.dump(self.tech_config, open(f"{self.output_folder}pv_wind_tech_configs.yaml", 'w'))

        resite_output = [
            self.formulation,
            self.timestamps,
            self.regions,
            self.modelling,
            self.technologies,
            self.deployment_vector,
            self.tech_points_dict,
            self.cap_potential_ds,
            self.cap_factor_df,
            self.existing_capacity_ds,
            self.optimal_capacity_ds,
            self.selected_tech_points_dict,
            self.tech_points_dict,
            self.generation_potential_df,
            self.load_df,
            self.selected_capacity_potential_ds,
            self.selected_cap_factor_df
        ]

        pickle.dump(resite_output, open(join(self.output_folder, 'resite_model.p'), 'wb'))



