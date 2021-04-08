from resite.resite import Resite

import pandas as pd
import numpy as np
import pypsa

from iepy.technologies import get_config_values
from iepy.geographics import match_points_to_regions, get_subregions
from iepy.technologies import get_costs

import logging

logger = logging.getLogger(__name__)


def timeseries_downsampling(net: pypsa.Network, sampling_rate: int):
    """
    Downsampling time series data. Load-driven method, i.e., first we resample load data and retrieve i) the median
    and ii) the index associated to the median; second, for other time series, we take values associated to the
    previously preserved indices.
    Parameters
    ----------
    net: pypsa.Network
        A PyPSA Network instance with buses associated to regions
    sampling_rate: int
    """

    periods = len(net.loads_t["p_set"].index) // sampling_rate
    idx = pd.date_range(net.loads_t["p_set"].index[0], net.loads_t["p_set"].index[-1], freq=f"{sampling_rate}h")
    load_ds_list = []
    gens_ds_list = []
    inflow_ds_list = []

    for c in net.loads.bus:
        median_values = []
        index_at_median_values = []

        load_at_bus = net.loads_t["p_set"].loc[:, "Load "+str(c)]
        for p in range(periods):
            load_at_bus_clip = load_at_bus.iloc[sampling_rate*p:sampling_rate*p+sampling_rate]
            median_values.append(load_at_bus_clip.median())
            index_at_median_values.append(load_at_bus_clip[load_at_bus_clip == load_at_bus_clip.median()].index[0])
        load_at_bus_resampled = pd.Series(data=median_values, index=idx)
        load_ds_list.append(load_at_bus_resampled.rename("Load "+str(c)))

        gens_at_bus = net.generators.index[net.generators.bus == c]
        gens_t_at_bus = net.generators_t["p_max_pu"].reindex(gens_at_bus, axis=1).dropna(axis=1, how='all')
        gens_ds_list.append(gens_t_at_bus.loc[index_at_median_values, :].set_index(idx))

        try:
            inflow_at_bus = net.storage_units_t["inflow"].loc[:, str(c) + " Storage reservoir"]
            inflow_at_bus = inflow_at_bus.loc[index_at_median_values]
            inflow_at_bus.index = idx
            inflow_ds_list.append(inflow_at_bus.rename(str(c) + " Storage reservoir"))
        except KeyError:
            continue

    net.loads_t["p_set"] = pd.concat(load_ds_list, axis=1)
    net.generators_t["p_max_pu"] = pd.concat(gens_ds_list, axis=1)
    net.storage_units_t["inflow"] = pd.concat(inflow_ds_list, axis=1)

    return net


def add_res_generators(net: pypsa.Network, config: dict, output_dir: str):
    """
    Adding RES generators to the PyPSA network.

    Parameters
    ----------
    net: pypsa.Network
        A PyPSA Network instance with buses associated to regions
    config: dict

    output_dir: str

    """

    countries = get_subregions(config['region'])
    technologies = config['res']['techs']
    logger.info(f"Adding RES {technologies} generation.")

    siting_params = config["res"]

    spatial_res = siting_params["spatial_resolution"]
    use_ex_cap = siting_params["use_ex_cap"]
    min_cap_pot = siting_params["min_cap_pot"]
    min_cap_if_sel = siting_params["min_cap_if_selected"]

    # Build sites for EU
    resite = Resite(countries, technologies, [net.snapshots[0], net.snapshots[-1]], spatial_res, min_cap_if_sel)
    regions_shapes = net.buses.loc[countries, ["onshore_region", 'offshore_region']]
    regions_shapes.columns = ['onshore', 'offshore']
    resite.build_data(use_ex_cap, min_cap_pot, regions_shapes=regions_shapes)
    net.cc_ds = resite.data_dict["capacity_credit_ds"]

    # Do siting if required
    if config["res"]["strategy"] == "siting":  # 'siting' case

        logger.info('resite model being built.')
        resite_output_folder = resite.init_output_folder(f"{output_dir}resite/")

        multipliers = compute_resite_input_shares(net, resite,
                                                temp_resolution=siting_params['formulation_params']['time_resolution'])
        siting_params['formulation_params'].update({'multiplier_per_region': multipliers.to_dict()})

        resite.build_model(siting_params["modelling"], siting_params['formulation'],
                           siting_params['formulation_params'],
                           siting_params['write_lp'], resite_output_folder)

        logger.info('Sending resite to solver.')
        resite.solve_model(resite_output_folder, solver=config['solver'], solver_options=config['solver_options'])

        logger.info("Saving resite results")
        resite.retrieve_selected_sites_data()
        resite.save(resite_output_folder)

        # Add solution to network
        logger.info('Retrieving resite results.')
        tech_location_dict = resite.sel_tech_points_dict
        existing_cap_ds = resite.sel_data_dict["existing_cap_ds"]
        cap_potential_ds = resite.sel_data_dict["cap_potential_ds"]
        cap_factor_df = resite.sel_data_dict["cap_factor_df"]

        if not resite.timestamps.equals(net.snapshots):
            # If network snapshots is a subset of resite snapshots just crop the data
            missing_timestamps = set(net.snapshots) - set(resite.timestamps)
            if not missing_timestamps:
                cap_factor_df = cap_factor_df.loc[net.snapshots]
            else:
                raise NotImplementedError("Network snapshots must currently be a subset of resite snapshots.")

    else:  # 'nositing' case
        tech_location_dict = resite.tech_points_dict
        existing_cap_ds = resite.data_dict["existing_cap_ds"]
        cap_potential_ds = resite.data_dict["cap_potential_ds"]
        cap_factor_df = resite.data_dict["cap_factor_df"]

    for tech, points in tech_location_dict.items():

        onshore_tech = get_config_values(tech, ['onshore'])

        # Associate sites to buses (using the associated shapes)
        buses = net.buses.copy()
        region_type = 'onshore_region' if onshore_tech else 'offshore_region'
        buses = buses.dropna(subset=[region_type])
        associated_buses = match_points_to_regions(points, buses[region_type]).dropna()
        points = list(associated_buses.index)

        p_nom_max = 'inf'
        if config['res']['limit_max_cap']:
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

    return net


def compute_resite_input_shares(net: pypsa.Network, resite: Resite, temp_resolution: str,
                                no_peak_load: int = 0.05, export_threshold: float = 0.7,
                                threshold_min: float = 0.1, threshold_max: float = 3.0):

    load_t = net.loads_t["p_set"]
    gens = net.generators
    gens_t = net.generators_t["p_max_pu"]
    sto_inflow = net.storage_units_t["inflow"]
    links = net.links[~net.links.index.str.contains('link')]
    buses = [bus[-2:] for bus in load_t.columns]

    exporting_bus_dict = {}
    multipliers = pd.Series(index=buses, dtype=float)

    dtau = temp_resolution[0].upper()

    for bus in buses:

        # Determine bus-specific load data, resample it at the temp discretization of the siting problem
        load_t_bus = load_t.loc[:, 'Load ' + bus].resample(dtau).sum().squeeze()

        # Determine CO2 free dispatchable legacy generation
        # RoR hydro
        ror_at_bus = gens[(gens.bus == bus) & (gens.type == 'ror')]
        if ror_at_bus.empty:
            ror_t_bus = pd.Series(np.zeros(len(load_t_bus.index)), index=load_t_bus.index)
        else:
            ror_gens_p_nom = ror_at_bus.p_nom_opt
            ror_gens_p_max_pu = gens_t.loc[:, ror_at_bus.index]
            ror_generation = ror_gens_p_max_pu * ror_gens_p_nom
            ror_t_bus = ror_generation.resample(dtau).sum().squeeze()

        # Nuclear
        nuke_at_bus = gens[(gens.bus == bus) & (gens.type == 'nuclear')]
        if nuke_at_bus.empty:
            nuke_t_bus = pd.Series(np.zeros(len(load_t_bus.index)), index=load_t_bus.index)
        else:
            nuke_gens_p_nom = nuke_at_bus.p_nom_opt.sum()
            nuke_generation = pd.Series(nuke_gens_p_nom, index=load_t.index)
            nuke_t_bus = nuke_generation.resample(dtau).sum().squeeze()

        # Reservoir hydro
        sto_gen_index = [c for c in sto_inflow.columns if bus in c and 'reservoir' in c]
        if not sto_gen_index:
            sto_t_bus = pd.Series(np.zeros(len(load_t_bus.index)), index=load_t_bus.index)
        else:
            sto_t_bus = sto_inflow.loc[:, sto_gen_index[0]].resample(dtau).sum().squeeze()

        # Calculate residual demand
        residual_t_bus = load_t_bus - (ror_t_bus + sto_t_bus + nuke_t_bus)
        # Retrieve indices of top y% instances and slice residual demand data
        peak_indices = residual_t_bus.nlargest(int(len(residual_t_bus.index) * no_peak_load)).index
        residual_t_bus = residual_t_bus[peak_indices]

        # Retrieve RES potential at bus and during the peak residual demand times
        res_gens = resite.tech_points_regions_ds[resite.tech_points_regions_ds == bus].index
        res_gens_pu_at_bus = resite.data_dict['cap_factor_df'].loc[:, res_gens]
        res_gens_p_max_at_bus = resite.data_dict['cap_potential_ds'].loc[res_gens]
        res_generation_at_bus = res_gens_pu_at_bus * res_gens_p_max_at_bus
        res_t_bus = res_generation_at_bus.resample(dtau).sum().sum(axis=1).squeeze()
        res_t_bus = res_t_bus[peak_indices]

        # Determine whether the bus is exporting or importing (will affect the contribution of transmission)
        load_bus_hourly = load_t.loc[:, 'Load ' + bus]
        res_bus_hourly = res_generation_at_bus.sum(axis=1)
        export_condition = (res_bus_hourly > load_bus_hourly).astype(int).sum()/len(load_t.index)
        exporting_bus_dict.update({bus: export_condition >= export_threshold})

        # Identify the links connected to this bus
        bus_links = links[(links.bus0 == bus) | (links.bus1 == bus)]
        # Compute a "theoretical" exchange potential
        # (installed capacity x nr of hours during dtau x expansion multiplier)
        dtau_h = load_t_bus.index.to_series().diff().astype('timedelta64[h]').iloc[-1]
        bus_links_max = bus_links.p_nom_max.sum() * dtau_h

        # Add the "export potential" to the bus-specific residual demand
        if exporting_bus_dict[bus]:
            residual_t_bus += bus_links_max
        else:
            residual_t_bus -= bus_links_max

        # Determine the multipliers (by keeping the median value in the timeseries)
        multipliers[bus] = res_t_bus.divide(residual_t_bus).median()

    # Clip series to limit min and max multipliers
    multipliers = multipliers.clip(lower=threshold_min, upper=threshold_max)

    return multipliers
