from os.path import join, dirname, abspath, isdir
from os import makedirs
import yaml
import pickle
from typing import List, Dict
from time import strftime
import importlib.util

import pandas as pd

from shapely.ops import unary_union

from pyggrid.data.generation.vres.legacy import get_legacy_capacity_in_regions
from pyggrid.data.generation.vres.profiles import compute_capacity_factors
from pyggrid.data.generation.vres.potentials.glaes import get_capacity_potential_for_shapes
from pyggrid.data.load import get_load
from pyggrid.data.geographics import get_shapes, get_subregions, match_points_to_regions
from pyggrid.data.technologies import get_config_dict, get_config_values

from pyggrid.data.geographics.grid_cells import get_grid_cells
from pyggrid.resite.models.utils import write_lp_file, solve_model

import logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(asctime)s - %(message)s")
logger = logging.getLogger()


class Resite:
    """
    Tool allowing the selection of RES sites.

    """

    # Methods import from other submodules
    solve_model = solve_model

    def __init__(self, regions: List[str], technologies: List[str], timeslice: List[str], spatial_resolution: float):
        """
        Constructor

        Parameters
        ----------
        regions: List[str]
            List of regions in which we want to site.
        technologies: List[str]
            List of technologies for which we want to site.
        timeslice: List[str]
            List of 2 string containing starting and end date of the time horizon.
        spatial_resolution: float
            Spatial resolution at which we want to site.
        """

        self.technologies = technologies
        self.regions = regions
        self.timestamps = pd.date_range(timeslice[0], timeslice[1], freq='1H')
        self.spatial_res = spatial_resolution

        self.instance = None
        self.data_dict = {}
        self.sel_data_dict = {}

        self.run_start = strftime('%Y%m%d_%H%M%S')

    def __str__(self):
        return f"Resite instance\n" \
               f"---------------\n" \
               f"Scope:\n" \
               f" - technologies: {self.technologies}\n" \
               f" - regions: {self.regions}\n" \
               f" - timeslice: {[str(self.timestamps[0]), str(self.timestamps[-1])]}\n" \
               f" - spatial resolution: {self.spatial_res}"

    def init_output_folder(self, output_folder: str = None) -> str:
        """Initialize an output folder."""

        if output_folder is None:
            output_folder = join(dirname(abspath(__file__)), f"../../output/resite/{self.run_start}/")
        assert output_folder[-1] == "/", "Error: Output folder name must end with '/'"
        if not isdir(output_folder):
            makedirs(output_folder)

        logger.info(f"Output folder path is: {abspath(output_folder)}/")

        return output_folder

    def build_data(self, use_ex_cap: bool, cap_pot_thresholds: List[float] = None):
        """Preprocess data.

        Parameters:
        -----------
        use_ex_cap: bool
            Whether to compute or not existing capacity and use it in optimization.
        cap_pot_thresholds: List[float] (default: None)
            List of thresholds per technology. Points with capacity potential under this threshold will be removed.
        """
        # TODO: this function needs to take as argument a vector data specifying which data it must compute

        # Compute total load (in GWh) for each region
        load_df = get_load(timestamps=self.timestamps, regions=self.regions, missing_data='interpolate')

        # Get shape of regions and list of subregions
        onshore_technologies = [get_config_values(tech, ["onshore"]) for tech in self.technologies]
        regions_shapes = pd.DataFrame(columns=["onshore", "offshore"], index=self.regions)
        all_subregions = []
        for region in self.regions:
            subregions = get_subregions(region)
            all_subregions.extend(subregions)
            shapes = get_shapes(subregions, save=True)
            if any(onshore_technologies):
                regions_shapes.loc[region, "onshore"] = unary_union(shapes[~shapes['offshore']]['geometry'])
            if not all(onshore_technologies):
                regions_shapes.loc[region, "offshore"] = unary_union(shapes[shapes['offshore']]['geometry'])

        # Divide the union of all regions shapes into grid cells of a given spatial resolution
        # TODO: this is shitty because you cannot add different technologies in separate regions
        onshore_union = unary_union(regions_shapes["onshore"]) if any(onshore_technologies) else None
        offshore_union = unary_union(regions_shapes["offshore"]) if not all(onshore_technologies) else None
        grid_cells_ds = get_grid_cells(self.technologies, self.spatial_res, onshore_union, offshore_union)

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
            for tech in self.technologies:
                tech_existing_cap_ds = \
                    get_legacy_capacity_in_regions(tech, grid_cells_ds.loc[tech].reset_index(drop=True),
                                                   all_subregions, raise_error=False)
                existing_cap_ds[tech] = tech_existing_cap_ds.values

        # Update capacity potential if existing capacity is bigger
        underestimated_capacity_indexes = existing_cap_ds > cap_potential_ds
        cap_potential_ds[underestimated_capacity_indexes] = existing_cap_ds[underestimated_capacity_indexes]

        # Remove sites that have a potential capacity under the desired value or equal to 0
        if cap_pot_thresholds is None:
            cap_pot_thresholds = [0]*len(self.technologies)
        assert len(cap_pot_thresholds) == len(self.technologies), \
            "Error: If you specify threshold on capacity potentials, you need to specify it for each technology."
        cap_pot_thresh_dict = dict(zip(self.technologies, cap_pot_thresholds))
        sites_to_drop = pd.DataFrame(cap_potential_ds).apply(lambda x: x[0] < cap_pot_thresh_dict[x.name[0]] or
                                                             x[0] == 0, axis=1)
        cap_potential_ds = cap_potential_ds[~sites_to_drop]
        existing_cap_ds = existing_cap_ds[~sites_to_drop]
        grid_cells_ds = grid_cells_ds[~sites_to_drop]

        # Compute capacity factors for each site
        tech_points_dict = {}
        techs = set(grid_cells_ds.index.get_level_values(0))
        for tech in techs:
            tech_points_dict[tech] = list(grid_cells_ds[tech].index)
        cap_factor_df = compute_capacity_factors(tech_points_dict, self.spatial_res, self.timestamps)

        # Associating coordinates to regions
        tech_points_regions_ds = pd.Series(index=grid_cells_ds.index)
        sites_index = tech_points_regions_ds.index
        for tech in set(sites_index.get_level_values(0)):
            on_off = 'onshore' if get_config_values(tech, ['onshore']) else 'offshore'
            tech_sites_index = sites_index[sites_index.get_level_values(0) == tech]
            points = list(zip(tech_sites_index.get_level_values(1), tech_sites_index.get_level_values(2)))
            tech_points_regions_ds[tech] = match_points_to_regions(points, regions_shapes[on_off]).values

        # Save all data in object
        self.use_ex_cap = use_ex_cap
        self.cap_pot_thresh_dict = cap_pot_thresh_dict
        self.tech_points_tuples = grid_cells_ds.index.values
        self.tech_points_dict = tech_points_dict
        self.initial_sites_ds = grid_cells_ds
        self.tech_points_regions_ds = tech_points_regions_ds
        self.data_dict["load"] = load_df
        self.data_dict["cap_potential_ds"] = cap_potential_ds
        self.data_dict["existing_cap_ds"] = existing_cap_ds
        self.data_dict["cap_factor_df"] = cap_factor_df

    def build_model(self, modelling: str, formulation: str, formulation_params: Dict,
                    write_lp: bool = False, output_folder: str = None):
        """
        Build model for the given formulation.

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

        # Check whether the formulation exists
        module_name = join(dirname(abspath(__file__)), f"models/{formulation}/")
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

    def retrieve_selected_sites_data(self):
        """Retrieve data for the selected sites and store it."""
        self.sel_tech_points_tuples = [(tech, point[0], point[1]) for tech, points in self.sel_tech_points_dict.items()
                                       for point in points]

        for name, data in self.data_dict.items():
            # Retrieve data only for series or dataframe containing per site data
            if (isinstance(data, pd.Series) and self.sel_tech_points_tuples[0] in data.index)\
                    or (isinstance(data, pd.DataFrame) and self.sel_tech_points_tuples[0] in data.columns):
                self.sel_data_dict[name] = data[self.sel_tech_points_tuples]

    def __getstate__(self):
        return (self.timestamps, self.regions, self.spatial_res, self.technologies,
                self.use_ex_cap, self.cap_pot_thresh_dict,
                self.formulation, self.formulation_params, self.modelling,
                self.tech_points_dict, self.data_dict, self.initial_sites_ds,
                self.sel_tech_points_tuples, self.sel_tech_points_dict,
                self.sel_data_dict, self.y_ds)

    def __setstate__(self, state):
        (self.timestamps, self.regions, self.spatial_res, self.technologies,
         self.use_ex_cap, self.cap_pot_thresh_dict,
         self.formulation, self.formulation_params, self.modelling,
         self.tech_points_dict, self.data_dict, self.initial_sites_ds,
         self.sel_tech_points_tuples, self.sel_tech_points_dict,
         self.sel_data_dict, self.y_ds) = state

    def save(self, dir_name: str = None):
        """Save all results and parameters."""

        output_folder = self.init_output_folder(dir_name)

        # Save some parameters to facilitate identification of run in directory
        params = {'spatial_resolution': self.spatial_res,
                  'timeslice': [str(self.timestamps[0]), str(self.timestamps[-1])],
                  'regions': self.regions,
                  'technologies': self.technologies,
                  'use_ex_cap': self.use_ex_cap,
                  'cap_pot_thresh_dict': self.cap_pot_thresh_dict,
                  'modelling': self.modelling,
                  'formulation': self.formulation,
                  'formulation_params': self.formulation_params}
        yaml.dump(params, open(f"{output_folder}config.yaml", 'w'))

        # Save the technology configurations
        yaml.dump(get_config_dict(self.technologies), open(f"{output_folder}tech_config.yaml", 'w'))

        # Save the attributes
        pickle.dump(self, open(f"{output_folder}resite_instance.p", 'wb'))
