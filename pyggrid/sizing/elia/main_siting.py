from os.path import isdir
from os import makedirs
from time import strftime

from pyomo.opt import ProblemFormat
import argparse

from pyggrid.data.indicators.emissions import get_reference_emission_levels_for_region
from pyggrid.data.topologies.tyndp2018 import get_topology
from pyggrid.data.geographics import get_subregions
from pyggrid.data.load import get_load
from pyggrid.data.technologies import get_config_dict
from pyggrid.network import *
from pyggrid.postprocessing.results_display import *
from pyggrid.sizing.elia.utils import upgrade_topology
from pyggrid.network.globals.functionalities_nopyomo \
    import add_extra_functionalities as add_extra_functionalities_nopyomo
from pyggrid.data.technologies import get_config_values
from shapely.ops import unary_union
from pyggrid.data.geographics import get_shapes, match_points_to_regions
from pyggrid.data.geographics.grid_cells import get_grid_cells
from pyggrid.data.generation.vres.potentials.glaes import get_capacity_potential_for_shapes
from pyggrid.data.generation.vres.legacy import get_legacy_capacity_in_regions
from pyggrid.data.generation.vres.profiles import compute_capacity_factors

import logging
logging.basicConfig(level=logging.DEBUG, format=f"%(levelname)s %(name) %(asctime)s - %(message)s")
logger = logging.getLogger(__name__)

NHoursPerYear = 8760.


def parse_args():

    parser = argparse.ArgumentParser(description='Command line arguments.')

    parser.add_argument('-sr', '--spatial_res', type=float, help='Spatial resolution')
    parser.add_argument('-pd', '--power_density', type=float, help='Offshore power density')
    parser.add_argument('-dir', '--resite_dir', type=str, help='resite directory')
    parser.add_argument('-fn', '--resite_fn', type=str, help='resite file name')
    parser.add_argument('-th', '--threads', type=int, help='Number of threads', default=1)

    parsed_args = vars(parser.parse_args())

    return parsed_args


if __name__ == '__main__':

    args = parse_args()
    logger.info(args)

    # Main directories
    data_dir = join(dirname(abspath(__file__)), "../../../data/")
    tech_dir = join(dirname(abspath(__file__)), "../../../data/technologies/")
    output_dir = join(dirname(abspath(__file__)), f"../../../output/sizing/elia/{strftime('%Y%m%d_%H%M%S')}/")

    # Run config
    config_fn = join(dirname(abspath(__file__)), 'config.yaml')
    config = yaml.load(open(config_fn, 'r'), Loader=yaml.FullLoader)

    solver_options = config["solver_options"][config["solver"]]
    if config["solver"] == 'gurobi':
        config["solver_options"][config["solver"]]['Threads'] = args['threads']
    else:
        config["solver_options"][config["solver"]]['threads'] = args['threads']

    # Parameters
    tech_info = pd.read_excel(join(tech_dir, 'tech_info.xlsx'), sheet_name='values', index_col=0)
    fuel_info = pd.read_excel(join(tech_dir, 'fuel_info.xlsx'), sheet_name='values', index_col=0)
    # tech_config = yaml.load(open(join(tech_dir, 'tech_config.yml')), Loader=yaml.FullLoader)

    # Compute and save results
    if not isdir(output_dir):
        makedirs(output_dir)

    # Save config and parameters files
    yaml.dump(config, open(f"{output_dir}config.yaml", 'w'), sort_keys=False)
    yaml.dump(get_config_dict(), open(f"{output_dir}tech_config.yaml", 'w'), sort_keys=False)
    tech_info.to_csv(f"{output_dir}tech_info.csv")
    fuel_info.to_csv(f"{output_dir}fuel_info.csv")

    # Time
    timeslice = config['time']['slice']
    time_resolution = config['time']['resolution']
    timestamps = pd.date_range(timeslice[0], timeslice[1], freq=f"{time_resolution}H")

    # Building network
    # Add location to Generators and StorageUnits
    override_comp_attrs = pypsa.descriptors.Dict({k: v.copy() for k, v in pypsa.components.component_attrs.items()})
    override_comp_attrs["Generator"].loc["x"] = ["float", np.nan, np.nan, "x in position (x;y)", "Input (optional)"]
    override_comp_attrs["Generator"].loc["y"] = ["float", np.nan, np.nan, "y in position (x;y)", "Input (optional)"]
    override_comp_attrs["StorageUnit"].loc["x"] = ["float", np.nan, np.nan, "x in position (x;y)", "Input (optional)"]
    override_comp_attrs["StorageUnit"].loc["y"] = ["float", np.nan, np.nan, "y in position (x;y)", "Input (optional)"]

    net = pypsa.Network(name="ELIA network (with siting)", override_component_attrs=override_comp_attrs)
    net.set_snapshots(timestamps)

    # Adding carriers
    for fuel in fuel_info.index[1:-1]:
        net.add("Carrier", fuel, co2_emissions=fuel_info.loc[fuel, "CO2"])

    # Loading topology
    logger.info("Loading topology.")
    eu_countries = get_subregions(config["region"])
    if config["add_TR"]:
        countries = eu_countries + ["TR"]
    net = get_topology(net, eu_countries, extend_line_cap=True)

    # Adding load
    logger.info("Adding load.")
    # onshore_bus_indexes = net.buses[net.buses.onshore].index
    load = get_load(timestamps=timestamps, countries=eu_countries, missing_data='interpolate')
    load_indexes = "Load " + pd.Index(eu_countries)
    loads = pd.DataFrame(load.values, index=net.snapshots, columns=load_indexes)
    net.madd("Load", load_indexes, bus=eu_countries, p_set=loads)

    if config["functionalities"]["load_shed"]["include"]:
        logger.info("Adding load shedding generators.")
        net = add_load_shedding(net, loads)

    # Add conventional gen
    if config["dispatch"]["include"]:
        tech = config["dispatch"]["tech"]
        net = add_conventional(net, tech)

    # Adding nuclear
    if config["nuclear"]["include"]:
        net = add_nuclear(net, eu_countries, config["nuclear"]["use_ex_cap"], config["nuclear"]["extendable"])

    if config["sto"]["include"]:
        net = add_sto_plants(net, 'countries', config["sto"]["extendable"], config["sto"]["cyclic_sof"])

    if config["phs"]["include"]:
        net = add_phs_plants(net, 'countries', config["phs"]["extendable"], config["phs"]["cyclic_sof"])

    if config["ror"]["include"]:
        net = add_ror_plants(net, 'countries', config["ror"]["extendable"])

    if config["battery"]["include"]:
        net = add_batteries(net, config["battery"]["type"], eu_countries)

    # Adding non-European nodes
    non_eu_res = config["non_eu"]["res"]
    if non_eu_res is not None:
        net = upgrade_topology(net, list(non_eu_res.keys()))
        # Add storage
        for region in non_eu_res.keys():
            if region in ["NA", "ME"]:
                neigh_countries = get_subregions(region)
            else:
                neigh_countries = [region]
            net = add_batteries(net, config["battery"]["type"], neigh_countries)

    # Adding pv and wind generators
    if config['res']['include']:
        for strategy, eu_technologies in config['res']['strategies'].items():
            # If no technology is associated to this strategy, continue
            if not len(eu_technologies):
                continue

            logger.info(f"Adding RES {eu_technologies} generation with strategy {strategy}.")

            # TODO: this could be integrated directly in resite
            # Creating grid cells for main region
            onshore_technologies = [get_config_values(tech, ["onshore"]) for tech in eu_technologies]
            # Divide the union of all regions shapes into grid cells of a given spatial resolution
            shapes = get_shapes(eu_countries, save=True)
            onshore_union = unary_union(shapes[~shapes['offshore']]['geometry']) if any(onshore_technologies) else None
            offshore_union = unary_union(shapes[shapes['offshore']]['geometry']) if not all(onshore_technologies) else None
            spatial_res = config["res"]["spatial_resolution"]
            grid_cells_ds = get_grid_cells(eu_technologies, spatial_res, onshore_union, offshore_union)

            # Compute grid_cells for non-eu
            out_techs = []
            out_countries = []
            for region in non_eu_res.keys():
                if region in ["NA", "ME"]:
                    neigh_countries = get_subregions(region)
                else:
                    neigh_countries = [region]
                out_countries += neigh_countries
                res_techs = non_eu_res[region]
                out_techs += res_techs
                # Creating grid cells for main region
                onshore_technologies = [get_config_values(tech, ["onshore"]) for tech in res_techs]
                # Divide the union of all regions shapes into grid cells of a given spatial resolution
                shapes = get_shapes(neigh_countries, save=True)
                onshore_union = unary_union(shapes[~shapes['offshore']]['geometry']) if any(
                    onshore_technologies) else None
                offshore_union = unary_union(shapes[shapes['offshore']]['geometry']) if not all(
                    onshore_technologies) else None
                spatial_res = config["res"]["spatial_resolution"]
                grid_cells_out_ds = get_grid_cells(res_techs, spatial_res, onshore_union, offshore_union)

                # Combine the series
                grid_cells_ds = pd.concat([grid_cells_ds, grid_cells_out_ds])

            # Compute data for grid cells
            all_techs = eu_technologies + out_techs
            all_countries = eu_countries + out_countries

            # Compute capacity factors for each site
            logging.info("Computing capacity factors")
            tech_points_dict = {}
            techs = set(grid_cells_ds.index.get_level_values(0))
            print(techs)
            for tech in techs:
                tech_points_dict[tech] = list(grid_cells_ds[tech].index)
            cap_factor_df = compute_capacity_factors(tech_points_dict, spatial_res, timestamps)

            # Compute capacities potential
            logging.info("Computing potential capacity")
            tech_config = get_config_dict(all_techs, ['filters', 'power_density'])
            cap_potential_ds = pd.Series(index=grid_cells_ds.index)
            for tech in all_techs:
                cap_potential_ds[tech] = \
                    get_capacity_potential_for_shapes(grid_cells_ds[tech].values, tech_config[tech]["filters"],
                                                      tech_config[tech]["power_density"])

            # Compute legacy capacity (only in main region, don't loose much in other regions)
            logging.info("Computing legacy capacity")
            existing_cap_ds = pd.Series(0., index=cap_potential_ds.index)
            use_ex_cap = config["res"]["use_ex_cap"]
            if use_ex_cap:
                for tech in eu_technologies:
                    tech_existing_cap_ds = \
                        get_legacy_capacity_in_regions(tech, grid_cells_ds.loc[tech].reset_index(drop=True),
                                                       eu_countries, raise_error=False)
                    existing_cap_ds[tech] = tech_existing_cap_ds.values

            # Update capacity potential if existing capacity is bigger
            underestimated_capacity_indexes = existing_cap_ds > cap_potential_ds
            cap_potential_ds[underestimated_capacity_indexes] = existing_cap_ds[underestimated_capacity_indexes]

            # Build a resite object
            from pyggrid.resite.resite import Resite
            resite = Resite([config["region"]], all_techs, config['time']['slice'], spatial_res)
            # TODO: change
            resite.data_dict["load"] = net.loads

            resite.data_dict["cap_potential_ds"] = cap_potential_ds
            resite.data_dict["existing_cap_ds"] = existing_cap_ds
            resite.data_dict["cap_factor_df"] = cap_factor_df
            resite.tech_points_tuples = grid_cells_ds.index.values
            resite.tech_points_dict = tech_points_dict
            resite.use_ex_cap = use_ex_cap
            resite.initial_sites_ds = grid_cells_ds
            resite.tech_points_regions_ds = pd.Series(config["region"], index=grid_cells_ds.index)
            resite.cap_pot_thresh_dict = None

            logger.info('resite model being built.')
            siting_params = config['res']
            resite.build_model(siting_params["modelling"], siting_params['formulation'],
                               siting_params['formulation_params'],
                               siting_params['write_lp'], output_dir)

            logger.info('Sending resite to solver.')
            resite.solve_model()

            # Add solution to network
            logger.info('Retrieving resite results.')
            tech_location_dict = resite.sel_tech_points_dict
            resite.retrieve_selected_sites_data()
            existing_cap_ds = resite.sel_data_dict["existing_cap_ds"]
            cap_potential_ds = resite.sel_data_dict["cap_potential_ds"]
            cap_factor_df = resite.sel_data_dict["cap_factor_df"]

            logger.info("Saving resite results")
            resite.save(output_dir)

            # TODO: for now they are always equal
            if not resite.timestamps.equals(net.snapshots):
                # If network snapshots is a subset of resite snapshots just crop the data
                missing_timestamps = set(net.snapshots) - set(resite.timestamps)
                if not missing_timestamps:
                    cap_factor_df = cap_factor_df.loc[net.snapshots]
                else:
                    # In other case, need to recompute capacity factors
                    raise NotImplementedError(
                        "Error: Network snapshots must currently be a subset of resite snapshots.")

            for tech, points in tech_location_dict.items():

                # Associate sites to buses
                onshore_tech = get_config_values(tech, ['onshore'])
                regions_ds = net.buses.region
                if not onshore_tech:
                    regions_ds = get_shapes(list(net.buses.index), which="offshore", save=True)["geometry"]
                associated_buses = match_points_to_regions(points, regions_ds).dropna()
                points = list(associated_buses.index)

                p_nom_max = 'inf'
                if config["res"]["limit_max_cap"]:
                    p_nom_max = cap_potential_ds[tech][points].values
                p_nom = existing_cap_ds[tech][points].values
                p_max_pu = cap_factor_df[tech][points].values

                capital_cost, marginal_cost = get_costs(tech, len(net.snapshots))

                net.madd("Generator",
                         pd.Index([f"Gen {tech} {x}-{y}" for x, y in points]),
                         bus=associated_buses.values,
                         p_nom_extendable=True,
                         p_nom_max=p_nom_max,
                         p_nom=p_nom,
                         p_nom_min=p_nom,
                         p_min_pu=0.,
                         p_max_pu=p_max_pu,
                         type=tech,
                         x=[x for x, _ in points],
                         y=[y for _, y in points],
                         marginal_cost=marginal_cost,
                         capital_cost=capital_cost)

    net.lopf(solver_name=config["solver"],
             solver_logfile=f"{output_dir}solver.log",
             solver_options=config["solver_options"][config["solver"]],
             extra_functionality=add_extra_functionalities,
             pyomo=True)

    if config['keep_lp']:
        net.model.write(filename=join(output_dir, 'model.lp'),
                        format=ProblemFormat.cpxlp,
                        # io_options={'symbolic_solver_labels': True})
                        io_options={'symbolic_solver_labels': False})
        net.model.objective.pprint()

    net.export_to_csv_folder(output_dir)
