from os.path import join, dirname, abspath, isdir
from os import makedirs
import yaml
import pickle
from typing import List, Dict, Tuple
from time import strftime
import importlib.util

import pandas as pd

from shapely.ops import unary_union
from shapely.geometry import MultiPoint

from src.data.legacy import get_legacy_capacity_in_regions
from src.data.vres_profiles import compute_capacity_factors
from src.data.vres_potential import get_capacity_potential_for_shapes
from src.data.load import get_load
from src.data.geographics import get_shapes, get_subregions
from src.data.technologies import get_config_dict

from src.resite.grid_cells import get_grid_cells
from src.resite.formulations.utils import write_lp_file, \
    solve_model, retrieve_solution as retrieve_solution_

import logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(asctime)s - %(message)s")
logger = logging.getLogger()


# TODO: maybe we should change the word point to 'cell' or 'site'?

class Resite:
    """
    Tool allowing the selection of RES sites.

    Methods
    -------
    __init__
    __del__
    init_output_folder
    build_input_data
    build_model
    solve_model
    retrieve_solution
    retrieve_sites_data
    save



    """

    # Methods import from other submodules
    solve_model = solve_model

    def __init__(self, regions: List[str], technologies: List[str], timeslice: List[str], spatial_resolution: float):
        """
        Constructor

        Parameters
        ----------
        regions: List[str]
            List of regions in which we want to site
        technologies: List[str]
            List of technologies for which we want to site
        timeslice: List[str]
            List of 2 string containing starting and end date of the time horizon
        spatial_resolution: float
            Spatial resolution at which we want to site
        """

        self.technologies = technologies
        self.regions = regions
        self.timestamps = pd.date_range(timeslice[0], timeslice[1], freq='1H')
        self.spatial_res = spatial_resolution

        self.instance = None

        self.run_start = strftime('%Y%m%d_%H%M%S')

    def init_output_folder(self, output_folder: str = None) -> str:
        """Initialize an output folder."""

        if output_folder is None:
            output_folder = join(dirname(abspath(__file__)), f"../../output/resite/{self.run_start}/")
        assert output_folder[-1] == "/", "Error: Output folder name must end with '/'"
        if not isdir(output_folder):
            makedirs(output_folder)

        logger.info(f"Output folder path is: {abspath(output_folder)}/")

        return output_folder

    def build_input_data(self, use_ex_cap: bool):
        """Preprocess data.

        Parameters:
        -----------
        use_ex_cap: bool
            Whether to compute or not existing capacity and use it in optimization
        """

        # Compute total load (in GWh) for each region
        load_df = get_load(timestamps=self.timestamps, regions=self.regions, missing_data='interpolate')

        # Get shape of regions and list of subregions
        regions_shapes = pd.Series(index=self.regions)
        onshore_shapes = []
        offshore_shapes = []
        all_subregions = []
        for region in self.regions:
            subregions = get_subregions(region)
            all_subregions.extend(subregions)
            shapes = get_shapes(subregions, save=True)
            onshore_shapes.extend(shapes[~shapes['offshore']]['geometry'].values)
            offshore_shapes.extend(shapes[shapes['offshore']]['geometry'].values)
            # TODO: this is fucking slow for EU...
            regions_shapes[region] = unary_union(shapes['geometry'])
        # TODO: maybe in some cases we could pass this directly?
        onshore_shape = unary_union(onshore_shapes)
        offshore_shape = unary_union(offshore_shapes)

        # Divide the union of all regions shapes into grid cells of a given spatial resolution
        grid_cells_ds = get_grid_cells(self.technologies, self.spatial_res, onshore_shape, offshore_shape)

        # Compute capacities potential
        tech_config = get_config_dict(self.technologies, ['filters', 'power_density'])
        cap_potential_ds = pd.Series(index=grid_cells_ds.index)
        for tech in self.technologies:
            cap_potential_ds[tech] = \
                get_capacity_potential_for_shapes(grid_cells_ds[tech].values, tech_config[tech]["filters"],
                                                  tech_config[tech]["power_density"])

        # Compute legacy capacity
        existing_cap_ds = pd.Series(0., index=cap_potential_ds.index)
        if use_ex_cap:
            # Get existing capacity at initial points, for technologies for which we can compute legacy data
            techs_with_legacy_data = list(set(self.technologies).intersection(['wind_onshore', 'wind_offshore',
                                                                               'pv_utility', 'pv_residential']))
            for tech in techs_with_legacy_data:
                tech_existing_cap_ds = \
                    get_legacy_capacity_in_regions(tech, grid_cells_ds.loc[tech].reset_index(drop=True),
                                                   all_subregions)
                existing_cap_ds[tech] = tech_existing_cap_ds.values

        # Update capacity potential if existing capacity is bigger
        underestimated_capacity_indexes = existing_cap_ds > cap_potential_ds
        cap_potential_ds[underestimated_capacity_indexes] = existing_cap_ds[underestimated_capacity_indexes]

        # Remove points that have a potential capacity under the desired value or equal to 0
        # TODO: this should be passed as an argument
        # TODO: if we do that though, shouldn't we put that also as a limit of minimum installable capacity per grid cell?
        potential_cap_thresholds = {tech: 0.01 for tech in self.technologies}
        points_to_drop = pd.DataFrame(cap_potential_ds).apply(lambda x:
                                                              x[0] < potential_cap_thresholds[x.name[0]] or x[0] == 0,
                                                              axis=1)
        cap_potential_ds = cap_potential_ds[~points_to_drop]
        existing_cap_ds = existing_cap_ds[~points_to_drop]
        grid_cells_ds = grid_cells_ds[~points_to_drop]

        # Compute capacity factors for each point
        tech_points_dict = {}
        techs = set(existing_cap_ds.index.get_level_values(0))
        for tech in techs:
            tech_points_dict[tech] = list(existing_cap_ds[tech].index)
        cap_factor_df = compute_capacity_factors(tech_points_dict, self.spatial_res, self.timestamps)

        # Associating coordinates to regions
        region_tech_points_dict = {region: set() for region in self.regions}
        for tech, points in tech_points_dict.items():
            points = MultiPoint(points)
            for region in self.regions:
                points_in_region = points.intersection(regions_shapes[region])
                points_in_region = [(tech, (point.x, point.y)) for point in points_in_region] \
                    if isinstance(points_in_region, MultiPoint) \
                    else [(tech, (points_in_region.x, points_in_region.y))]
                region_tech_points_dict[region] = region_tech_points_dict[region].union(set(points_in_region))

        # Save all data in object
        self.use_ex_cap = use_ex_cap
        self.tech_points_tuples = grid_cells_ds.index.values
        self.tech_points_dict = tech_points_dict
        self.region_tech_points_dict = region_tech_points_dict
        self.initial_sites_ds = grid_cells_ds
        self.load_df = load_df
        self.cap_potential_ds = cap_potential_ds
        self.existing_cap_ds = existing_cap_ds
        self.cap_factor_df = cap_factor_df

    def build_model(self, modelling: str, formulation: str, formulation_params: Dict,
                    write_lp: bool = False, output_folder: str = None):
        """
        Model build-up.

        Parameters:
        ------------
        modelling: str
            Choice of modelling language
        formulation: str
            Formulation of the optimization problem to solve
        formulation_params: Dict
            Parameters need by the formulation.
        write_lp : bool (default: False)
            If True, the model is written to an .lp file.
        dir_name: str (default: None)
            Where to write the .lp file
        """

        if formulation in ['meet_RES_targets_agg', 'meet_RES_targets_hourly', 'meet_RES_targets_daily',
                            'meet_RES_targets_weekly', 'meet_RES_targets_monthly', 'maximize_generation',
                            'maximize_aggr_cap_factor'] and len(formulation_params) != len(self.regions):
            raise ValueError('For the selected formulation, the "regions" and "formulation_params" '
                             'lists must have the same cardinality!')

        # Check that formulation exists
        module_name = join(dirname(abspath(__file__)), f"formulations/{formulation}/")
        assert isdir(module_name), f"Error: No model exists for formulation {formulation}."

        # Load formulation module and execute
        spec = importlib.util.spec_from_file_location("module.name", f"{module_name}model.py")
        formulation_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(formulation_module)
        formulation_module.build_model(self, modelling, formulation_params)

        if write_lp:
            output_folder = self.init_output_folder(output_folder)
            write_lp_file(self.instance, modelling, output_folder)

        self.modelling = modelling
        self.formulation = formulation
        self.formulation_params = formulation_params

    def retrieve_solution(self) -> Dict[str, List[Tuple[float, float]]]:
        """
        Get points that were selected during the optimization.

        Returns
        -------
        Dict[str, List[Tuple[float, float]]]
            Lists of selected points for each technology

        """
        self.selected_tech_points_dict, self.optimal_cap_ds = retrieve_solution_(self)

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

        output_folder = self.init_output_folder(dir_name)

        # Save some parameters to facilitate identification of run in directory
        params = {'spatial_resolution': self.spatial_res,
                  # 'filtering_layers': self.filtering_layers,
                  'timeslice': [str(self.timestamps[0]), str(self.timestamps[-1])],
                  'regions': self.regions,
                  'technologies': self.technologies,
                  'use_ex_cap': self.use_ex_cap,
                  'modelling': self.modelling,
                  'formulation': self.formulation,
                  'formulation_params': self.formulation_params}
        yaml.dump(params, open(f"{output_folder}config.yaml", 'w'))

        # Save the technology configurations
        yaml.dump(get_config_dict(self.technologies), open(f"{output_folder}tech_config.yaml", 'w'))

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
            # self.generation_potential_df,
            self.load_df,
            self.selected_cap_potential_ds,
            self.selected_cap_factor_df
        ]

        pickle.dump(resite_output, open(join(output_folder, 'resite_model.p'), 'wb'))
