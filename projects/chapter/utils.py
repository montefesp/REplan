import pypsa
import pandas as pd

from iepy.technologies import get_tech_info


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


def build_uc_instance(net: pypsa.Network, nfull: pypsa.Network):

    generators = net.generators
    gens_to_drop = generators[generators.p_nom_opt < 1e-3].index
    generators = generators.drop(gens_to_drop)
    commit = ['nuclear', 'ocgt', 'ccgt']
    generators['control'] = 'PQ'
    generators['p_nom'] = generators['p_nom_opt']
    generators['p_nom_extendable'] = False
    generators['up_time_before'] = 0
    for t in commit:
        gens_tech = generators[generators['type'] == t].copy()
        base_level = get_tech_info(t, ["base_level"]).values[0]
        for idx in gens_tech.index:
            gens_tech.at[idx, 'committable'] = True
            gens_tech.at[idx, 'p_min_pu'] = base_level
            gens_tech.at[idx, 'start_up_cost'] *= gens_tech.at[idx, 'p_nom_opt']
        generators[generators['type'] == t] = gens_tech

    stores = net.stores
    stores['e_nom'] = stores['e_nom_opt']
    stores['e_nom_extendable'] = False

    storage_units = net.storage_units
    storage_units['cyclic_state_of_charge'] = False
    storage_units['p_nom_extendable'] = False

    links = net.links
    links['p_nom'] = links['p_nom_opt']
    links['p_nom_extendable'] = False

    lines = net.lines
    lines['s_nom'] = lines['s_nom_opt']
    lines['s_nom_extendable'] = False

    nfull.generators = generators
    nfull.stores = stores
    nfull.links = links
    nfull.lines = lines
    nfull.storage_units = storage_units

    return nfull
