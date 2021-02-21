import pypsa


def timeseries_downsampling(net: pypsa.Network, sampling_rate: int):

    net.loads_t["p_set"] = net.loads_t["p_set"].resample(f"{sampling_rate}h").median()
    net.generators_t["p_max_pu"] = net.generators_t["p_max_pu"].resample(f"{sampling_rate}h").median()
    net.storage_units_t["inflow"] = net.storage_units_t["inflow"].resample(f"{sampling_rate}h").median()

    return net
