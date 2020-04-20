from os.path import join, dirname, abspath, isdir
from os import makedirs
import yaml
import pickle
from typing import List, Dict, Tuple, Any
from time import strftime

import pandas as pd

from shapely.ops import cascaded_union
from shapely.geometry import MultiPoint

from src.data.legacy import get_legacy_capacity_at_points
from src.data.resource import read_resource_database, compute_capacity_factors
from src.data.land_data import filter_points
from src.data.res_potential import get_capacity_potential_at_points
from src.data.load import get_load
from src.data.geographics import return_region_shape, return_points_in_shape, get_subregions

import logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(asctime)s - %(message)s")
logger = logging.getLogger()


def init_output_folder(output_folder: str = None) -> str:
    """Initialize an output folder."""

    if output_folder is None:
        output_folder = join(dirname(abspath(__file__)), f"../../output/resite/{strftime('%Y%m%d_%H%M%S')}/")
    assert output_folder[-1] == "/", "Error: Output folder name must end with '/'"
    if not isdir(output_folder):
        makedirs(output_folder)

    logger.info(f"Output folder path is: {abspath(output_folder)}/")

    return output_folder


class Resite:
    """
    Tool allowing the selection of RES sites.

    Methods
    -------
    __init__
    __del__
    build_input_data
    build_model
    solve_model
    retrieve_solution
    retrieve_sites_data
    save



    """

    def __init__(self, regions: List[str], technologies: List[str], tech_config: Dict[str, Any], timeslice: List[str],
                 spatial_resolution: float):
        """
        Constructor

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
        """

        self.technologies = technologies
        self.regions = regions
        self.tech_config = tech_config
        self.timestamps = pd.date_range(timeslice[0], timeslice[1], freq='1H')
        self.spatial_res = spatial_resolution

        self.instance = None

    def build_input_data(self, use_ex_cap: bool, filtering_layers: Dict[str, bool]):
        """Preprocess data.

        Parameters:
        -----------
        use_ex_cap: bool
            Whether to compute or not existing capacity and use it in optimization
        filtering_layers: Dict[str, bool]
            Dictionary indicating if a given filtering layers needs to be applied.
            If the layer name is present as key and associated to a True boolean,
            then the corresponding is applied.
        """

        self.use_ex_cap = use_ex_cap
        self.filtering_layers = filtering_layers

        # self.load_df = get_prepared_load(timestamps=self.timestamps, regions=self.regions)
        self.load_df = get_load(timestamps=self.timestamps, regions=self.regions, missing_data='interpolate')

        # Get shape of regions and list of subregions
        regions_shapes = pd.Series(index=self.regions)
        all_subregions = []
        for region in self.regions:
            subregions = get_subregions(region)
            all_subregions += subregions
            shapes = return_region_shape(region, subregions)
            regions_shapes[region] = cascaded_union([shapes['onshore'], shapes['offshore']])

        # Get all points situated in the given regions at the given spatial resolution
        # TODO: Need to remove the first init_points by downloading new data
        path_resource_data = join(dirname(abspath(__file__)),
                                  f"../../data/resource/source/era5-land/{self.spatial_res}")
        database = read_resource_database(path_resource_data)
        init_points = list(zip(database.longitude.values, database.latitude.values))
        init_points = return_points_in_shape(cascaded_union(regions_shapes.values), self.spatial_res, init_points)

        # Filter those points
        self.tech_points_dict = filter_points(self.technologies, self.tech_config, init_points, self.spatial_res,
                                              filtering_layers)

        if use_ex_cap:
            # Get existing capacity at initial points, for technologies for which we can compute legacy data
            tech_with_legacy_data = list(set(self.technologies).intersection(['wind_onshore', 'wind_offshore',
                                                                              'pv_utility', 'pv_residential']))
            existing_cap_ds = get_legacy_capacity_at_points(tech_with_legacy_data, self.tech_config,
                                                                 all_subregions, init_points, self.spatial_res)

            # If some initial points with existing capacity were filtered, add them back
            for tech in tech_with_legacy_data:
                if tech in existing_cap_ds.index.get_level_values(0):
                    self.tech_points_dict[tech] += list(existing_cap_ds[tech].index)
                # Remove duplicates
                self.tech_points_dict[tech] = list(set(self.tech_points_dict[tech]))

        # Remove techs that have no points associated to them
        self.tech_points_dict = {k: v for k, v in self.tech_points_dict.items() if len(v) > 0}

        # Create dataframe with existing capacity, for all points (not just the ones with existing capacity)
        self.tech_points_tuples = [(tech, point) for tech, points in self.tech_points_dict.items() for point in points]
        self.existing_cap_ds = pd.Series(0., index=pd.MultiIndex.from_tuples(self.tech_points_tuples))
        if use_ex_cap:
            self.existing_cap_ds.loc[existing_cap_ds.index] = existing_cap_ds.values

        # Compute capacity factors for each point
        self.cap_factor_df = compute_capacity_factors(self.tech_points_dict, self.tech_config,
                                                      self.spatial_res, self.timestamps)

        # Compute capacity potential for each point (taking into account existing capacity)
        self.cap_potential_ds = get_capacity_potential_at_points(self.tech_points_dict, self.spatial_res,
                                                                 all_subregions, self.existing_cap_ds)

        # Compute the percentage of potential capacity that is covered by existing capacity
        existing_cap_percentage_ds = self.existing_cap_ds.divide(self.cap_potential_ds)

        # Remove points which have zero potential capacity
        self.existing_cap_percentage_ds = existing_cap_percentage_ds.dropna()
        self.cap_potential_ds = self.cap_potential_ds[self.existing_cap_percentage_ds.index]
        self.cap_factor_df = self.cap_factor_df[self.existing_cap_percentage_ds.index]
        self.existing_cap_ds = self.existing_cap_ds[self.existing_cap_percentage_ds.index]

        # Retrieve final points
        self.tech_points_tuples = self.existing_cap_percentage_ds.index.values
        self.tech_points_dict = {}
        techs = set(self.existing_cap_percentage_ds.index.get_level_values(0))
        for tech in techs:
            self.tech_points_dict[tech] = list(self.existing_cap_ds[tech].index)

        # Maximum generation that can be produced if max capacity installed
        self.generation_potential_df = self.cap_factor_df * self.cap_potential_ds

        # Associating coordinates to regions
        self.region_tech_points_dict = {region: set() for region in self.regions}
        for tech, points in self.tech_points_dict.items():
            points = MultiPoint(points)
            for region in self.regions:
                points_in_region = points.intersection(regions_shapes[region])
                points_in_region = [(tech, (point.x, point.y)) for point in points_in_region] \
                    if isinstance(points_in_region, MultiPoint) \
                    else [(tech, (points_in_region.x, points_in_region.y))]
                self.region_tech_points_dict[region] = self.region_tech_points_dict[region].union(set(points_in_region))

    def build_model(self, modelling: str, formulation: str, formulation_params: List[float],
                    write_lp: bool = False, output_folder: str = None):
        """
        Model build-up.

        Parameters:
        ------------
        modelling: str
            Choice of modelling language
        formulation: str
            Formulation of the optimization problem to solve
        formulation_params: List[float]
            Each formulation requires a different set of parameters.
            For 'meet_RES_targets' formulations, the list must contain the percentage of load that must be met
            in each region.
            For 'meet_demand_with_capacity' formulation, the list must contain the capacity (in GW) that is required
            to be installed for each technology in the model.
            For 'maximize' formulations, the list must contain the number of sites to be deployed per region.
        write_lp : bool (default: False)
            If True, the model is written to an .lp file.
        dir_name: str (default: None)
            Where to write the .lp file
        """

        if formulation == 'meet_demand_with_capacity' and len(self.regions) != 1:
            raise ValueError('The selected formulation works for one region only!')
        elif formulation in ['meet_RES_targets_agg', 'meet_RES_targets_hourly', 'meet_RES_targets_daily',
                             'meet_RES_targets_weekly', 'meet_RES_targets_monthly', 'maximize_generation',
                             'maximize_aggr_cap_factor'] and len(formulation_params) != len(self.regions):
            raise ValueError('For the selected formulation, the "regions" and "formulation_params" '
                             'lists must have the same cardinality!')

        accepted_modelling = ['pyomo', 'docplex', 'gurobipy']
        assert modelling in accepted_modelling, f"Error: {modelling} is not available as modelling language. " \
                                                f"Accepted languages are {accepted_modelling}"

        if write_lp:
            output_folder = init_output_folder(output_folder)

        self.modelling = modelling
        self.formulation = formulation
        self.formulation_params = formulation_params
        if self.modelling == 'pyomo':
            from src.resite.models.pyomo import build_model as build_pyomo_model
            build_pyomo_model(self, formulation, formulation_params, write_lp, output_folder)
        elif self.modelling == 'docplex':
            from src.resite.models.docplex import build_model as build_docplex_model
            build_docplex_model(self, formulation, formulation_params, write_lp, output_folder)
        elif self.modelling == 'gurobipy':
            from src.resite.models.gurobipy import build_model as build_gurobipy_model
            build_gurobipy_model(self, formulation, formulation_params, write_lp, output_folder)

    def solve_model(self):
        """Solve the model built with build_model"""
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
        Get points that were selected during the optimization.

        Returns
        -------
        Dict[str, List[Tuple[float, float]]]
            Lists of selected points for each technology

        """
        if self.modelling == 'pyomo':
            from src.resite.models.pyomo import retrieve_solution as retrieve_pyomo_solution
            self.objective, self.selected_tech_points_dict, self.optimal_cap_ds = retrieve_pyomo_solution(self)
        elif self.modelling == 'docplex':
            from src.resite.models.docplex import retrieve_solution as retrieve_docplex_solution
            self.objective, self.selected_tech_points_dict, self.optimal_cap_ds = retrieve_docplex_solution(self)
        elif self.modelling == 'gurobipy':
            from src.resite.models.gurobipy import retrieve_solution as retrieve_gurobipy_solution
            self.objective, self.selected_tech_points_dict, self.optimal_cap_ds = retrieve_gurobipy_solution(self)

        return self.selected_tech_points_dict

    def retrieve_sites_data(self):
        """
        Return data for the optimal sites.

        Returns
        -------
        self.selected_existing_capacity_ds: pd.Series
            Pandas series giving for each (tech, coord) tuple in self.selected_tech_points_dict the existing
            cap at these positions
        self.selected_cap_potential_ds: pd.Series
            Pandas series giving for each (tech, coord) tuple in self.selected_tech_points_dict the capacity
            potential at these positions .
        self.selected_cap_factor_df: pd.DataFrame
            Pandas series indexed by time giving for each (tech, coord) tuple in self.selected_tech_points_dict
            its cap factors time series

        """

        selected_tech_points_tuples = [(tech, point) for tech, points in self.selected_tech_points_dict.items()
                                       for point in points]

        self.selected_existing_cap_ds = self.existing_cap_ds.loc[selected_tech_points_tuples]
        self.selected_cap_potential_ds = self.cap_potential_ds.loc[selected_tech_points_tuples]
        self.selected_cap_factor_df = self.cap_factor_df[selected_tech_points_tuples]

        return self.selected_existing_cap_ds, self.selected_cap_potential_ds, self.selected_cap_factor_df

    def save(self, dir_name: str = None):
        """Save all results and parameters."""

        output_folder = init_output_folder(dir_name)

        # Save some parameters to facilitate identification of run in directory
        params = {'spatial_resolution': self.spatial_res,
                  'filtering_layers': self.filtering_layers,
                  'timeslice': [str(self.timestamps[0]), str(self.timestamps[-1])],
                  'regions': self.regions,
                  'technologies': self.technologies,
                  'use_ex_cap': self.use_ex_cap,
                  'modelling': self.modelling,
                  'formulation': self.formulation,
                  'formulation_params': self.formulation_params}
        yaml.dump(params, open(f"{output_folder}config.yaml", 'w'))

        # Save the technology configurations
        yaml.dump(self.tech_config, open(f"{output_folder}pv_wind_tech_configs.yaml", 'w'))

        # Save the attributes
        resite_output = [
            self.formulation,
            self.timestamps,
            self.regions,
            self.modelling,
            self.use_ex_cap,
            self.spatial_res,
            self.technologies,
            self.formulation_params,
            self.tech_points_dict,
            self.cap_potential_ds,
            self.cap_factor_df,
            self.existing_cap_ds,
            self.optimal_cap_ds,
            self.selected_tech_points_dict,
            self.tech_points_dict,
            self.generation_potential_df,
            self.load_df,
            self.selected_cap_potential_ds,
            self.selected_cap_factor_df
        ]

        pickle.dump(resite_output, open(join(output_folder, 'resite_model.p'), 'wb'))



