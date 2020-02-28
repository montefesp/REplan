from os.path import join, dirname, abspath
from shapely.ops import cascaded_union
from shapely.geometry import MultiPoint
import pandas as pd
from src.resite.utils import custom_log, init_folder
from src.data.legacy.manager import get_legacy_capacity
from src.data.resource.manager import read_resource_database, compute_capacity_factors
from src.data.land_data.manager import filter_points
from src.data.res_potential.manager import get_capacity_potential
from src.data.load.manager import retrieve_load_data
from src.data.geographics.manager import return_region_shape, return_points_in_shape, get_subregions
from typing import List, Dict
from shutil import copy
import pickle
import yaml

from src.resite.models.pyomo import build_model as build_pyomo_model, solve_model as solve_pyomo_model


class Resite:

    build_pyomo_model = build_pyomo_model
    solve_pyomo_model = solve_pyomo_model

    def __init__(self, params):

        self.output_folder = init_folder(params['keep_files'])
        copy('config_model.yml', self.output_folder)
        copy('config_techs.yml', self.output_folder)

        tech_config_path = join(dirname(abspath(__file__)), 'config_techs.yml')
        self.tech_config = yaml.load(open(tech_config_path), Loader=yaml.FullLoader)

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
        database = read_resource_database(path_resource_data)
        init_points = list(zip(database.longitude.values, database.latitude.values))
        init_points = return_points_in_shape(regions_shapes_union, self.spatial_res, init_points)

        custom_log("Filtering coordinates")
        self.tech_points_dict = filter_points(self.technologies, self.tech_config, init_points, self.spatial_res,
                                              filtering_layers)

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
        self.tech_points_tuples = [(tech, point) for tech, points in self.tech_points_dict.items() for point in points]
        existing_capacity_ds = pd.Series(0., index=pd.MultiIndex.from_tuples(self.tech_points_tuples))
        for tech, coord in existing_capacity_ds.index:
            if tech in existing_capacity_dict and existing_capacity_dict[tech] is not None \
                    and coord in existing_capacity_dict[tech]:
                existing_capacity_ds[tech, coord] = existing_capacity_dict[tech][coord]

        custom_log("Compute cap factor")
        self.cap_factor_df = compute_capacity_factors(self.tech_points_dict, self.tech_config,
                                                      self.spatial_res, self.timestamps)

        custom_log("Compute capacity potential per node")
        self.cap_potential_ds = get_capacity_potential(self.tech_points_dict, self.spatial_res, self.regions,
                                                       existing_capacity_ds)

        # Compute percentage of existing capacity and set to 1. when capacity is zero
        existing_cap_percentage_ds = existing_capacity_ds.divide(self.cap_potential_ds)
        self.existing_cap_percentage_ds = existing_cap_percentage_ds.fillna(1.)

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
        output_folder: str
            Path towards output folder
        write_lp : bool (default: False)
            If True, the model is written to an .lp file.
        """

        accepted_modelling = ['pyomo']
        assert modelling in accepted_modelling, f"Error: {modelling} is not available as modelling language. " \
                                                f"Accepted languages are {accepted_modelling}"
        self.modelling = modelling
        if self.modelling == 'pyomo':
            self.build_pyomo_model(formulation, deployment_vector, write_lp)

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
        self.solve_pyomo_model(solver, solver_options)

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

