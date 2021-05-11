import pypsa
import pandas as pd


# def timeseries_downsampling(net: pypsa.Network, sampling_rate: int):
#     """
#     Downsampling time series data. Load-driven method, i.e., first we resample load data and retrieve i) the median
#     and ii) the index associated to the median; second, for other time series, we take values associated to the
#     previously preserved indices.
#
#     Parameters
#     ----------
#     net: pypsa.Network
#         A PyPSA Network instance with buses associated to regions
#     sampling_rate: int
#
#     """
#
#     periods = len(net.loads_t["p_set"].index) // sampling_rate
#     idx = pd.date_range(net.loads_t["p_set"].index[0], net.loads_t["p_set"].index[-1], freq=f"{sampling_rate}h")
#     load_ds_list = []
#     gens_ds_list = []
#     inflow_ds_list = []
#
#     for c in net.loads.bus:
#         median_values = []
#         index_at_median_values = []
#
#         load_at_bus = net.loads_t["p_set"].loc[:, "Load "+str(c)]
#         for p in range(periods):
#             load_at_bus_clip = load_at_bus.iloc[sampling_rate*p:sampling_rate*p+sampling_rate]
#             median_values.append(load_at_bus_clip.median())
#             index_at_median_values.append(load_at_bus_clip[load_at_bus_clip == load_at_bus_clip.median()].index[0])
#         load_at_bus_resampled = pd.Series(data=median_values, index=idx)
#         load_ds_list.append(load_at_bus_resampled.rename("Load "+str(c)))
#
#         gens_at_bus = net.generators.index[net.generators.bus == c]
#         gens_t_at_bus = net.generators_t["p_max_pu"].reindex(gens_at_bus, axis=1).dropna(axis=1, how='all')
#         gens_ds_list.append(gens_t_at_bus.loc[index_at_median_values, :].set_index(idx))
#
#         try:
#             inflow_at_bus = net.storage_units_t["inflow"].loc[:, str(c) + " Storage reservoir"]
#             inflow_at_bus = inflow_at_bus.loc[index_at_median_values]
#             inflow_at_bus.index = idx
#             inflow_ds_list.append(inflow_at_bus.rename(str(c) + " Storage reservoir"))
#         except KeyError:
#             continue
#
#     net.loads_t["p_set"] = pd.concat(load_ds_list, axis=1)
#     net.generators_t["p_max_pu"] = pd.concat(gens_ds_list, axis=1)
#     net.storage_units_t["inflow"] = pd.concat(inflow_ds_list, axis=1)
#
#     return net


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

    net.loads_t["p_set"] = net.loads_t["p_set"].resample(f"{sampling_rate}h").mean()
    net.generators_t["p_max_pu"] = net.generators_t["p_max_pu"].resample(f"{sampling_rate}h").mean()
    net.storage_units_t["inflow"] = net.storage_units_t["inflow"].resample(f"{sampling_rate}h").mean()

    return net


def compute_capacity_credit_ds(net: pypsa.Network, peak_sample: float = 0.01):
    """
    Compute capacity credit based on Milligan eq. from the CF dataframe of all candidate sites.

    Parameters:
    ------------
    net: pypsa.Network

    peak_sample : float (default: 0.01, 1%)
        The top % wrt which the capacity credit is computed.
    """

    # res_technologies = ['wind_onshore', 'wind_offshore', 'pv_utility', 'pv_residential']
    res_technologies = net.config['res']['strategies']['from_files']['which']
    if net.config['res']['strategies']['bus']['extendable']:
        res_technologies = list(set(res_technologies).union(net.config['res']['strategies']['bus']['which']))
    gens = net.generators[net.generators.type.str.contains('|'.join(res_technologies))]
    gens_t = net.generators_t.p_max_pu

    cc_ds = pd.Series(index=gens.index, dtype=float)
    nvals = int(peak_sample * len(net.loads_t.p_set.index))

    for c in net.loads_t.p_set.columns:
        load_data = net.loads_t.p_set.loc[:, c].squeeze()
        load_data_peak_index = load_data.nlargest(nvals).index

        gens_in_c = gens[gens.bus == c[-2:]].index
        cc_ds.loc[gens_in_c] = gens_t.loc[load_data_peak_index, gens_in_c].mean()

    net.cc_ds = cc_ds

    return net
