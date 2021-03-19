import pypsa
import pandas as pd


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
