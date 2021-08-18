import pypsa
import pandas as pd


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

        gens_in_c = gens[gens.bus == c.split(' ')[-1]].index
        cc_ds.loc[gens_in_c] = gens_t.loc[load_data_peak_index, gens_in_c].mean()

    net.cc_ds = cc_ds

    return net
