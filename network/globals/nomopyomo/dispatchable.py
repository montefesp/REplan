from typing import Dict

import pypsa
from pypsa.linopt import get_var, linexpr, define_constraints


def dispatchable_capacity_lower_bound(net: pypsa.Network, thresholds: Dict):
    """
    Constraint that ensures a minimum dispatchable installed capacity.

    Parameters
    ----------
    net: pypsa.Network
        A PyPSA Network instance with buses associated to regions

    thresholds: Dict
        Dict containing scalar thresholds for disp_capacity/peak_load for each bus
    """
    # TODO: extend for different topologies, if necessary
    dispatchable_technologies = ['ocgt', 'ccgt', 'ccgt_ccs', 'nuclear', 'sto']
    for bus in net.loads.bus:

        if bus in thresholds.keys():

            lhs = ''
            legacy_at_bus = 0

            gens = net.generators[(net.generators.bus == bus) & (net.generators.type.isin(dispatchable_technologies))]
            for gen in gens.index:
                if gens.loc[gen].p_nom_extendable:
                    lhs += linexpr((1., get_var(net, 'Generator', 'p_nom')[gen]))
                else:
                    legacy_at_bus += gens.loc[gen].p_nom_min

            stos = net.storage_units[(net.storage_units.bus == bus) &
                                     (net.storage_units.type.isin(dispatchable_technologies))]
            for sto in stos.index:
                if stos.loc[sto].p_nom_extendable:
                    lhs += linexpr((1., get_var(net, 'StorageUnit', 'p_nom')[sto]))
                else:
                    legacy_at_bus += stos.loc[sto].p_nom_min

            # Get load for country
            load_idx = net.loads[net.loads.bus == bus].index
            load_peak = net.loads_t.p_set[load_idx].max()

            load_peak_threshold = load_peak * thresholds[bus]
            rhs = max(0, load_peak_threshold.values[0] - legacy_at_bus)

            define_constraints(net, lhs.sum(), '>=', rhs, 'disp_capacity_lower_bound', bus)


def add_planning_reserve_constraint(net: pypsa.Network, prm: float):
    """
    Constraint that ensures a minimum dispatchable installed capacity.

    Parameters
    ----------
    net: pypsa.Network
        A PyPSA Network instance with buses associated to regions
    prm: float
        Planning reserve margin.
    """
    cc_ds = net.cc_ds
    dispatchable_technologies = ['ocgt', 'ccgt', 'ccgt_ccs', 'nuclear', 'sto']

    res_technologies = net.config['res']['strategies']['from_files']['which']
    if net.config['res']['strategies']['bus']['extendable']:
        res_technologies = list(set(res_technologies).union(net.config['res']['strategies']['bus']['which']))

    for bus in net.loads.bus:

        lhs = ''
        legacy_at_bus = 0

        gens = net.generators[(net.generators.bus == bus) & (net.generators.type.isin(dispatchable_technologies))]
        for gen in gens.index:
            if gens.loc[gen].p_nom_extendable:
                lhs += linexpr((1., get_var(net, 'Generator', 'p_nom')[gen]))
            else:
                legacy_at_bus += gens.loc[gen].p_nom_min

        stos = net.storage_units[(net.storage_units.bus == bus) &
                                 (net.storage_units.type.isin(dispatchable_technologies))]
        for sto in stos.index:
            if stos.loc[sto].p_nom_extendable:
                lhs += linexpr((1., get_var(net, 'StorageUnit', 'p_nom')[sto]))
            else:
                legacy_at_bus += stos.loc[sto].p_nom_min

        res_gens = net.generators[(net.generators.bus == bus) &
                                  (net.generators.type.str.contains('|'.join(res_technologies)))]
        for gen in res_gens.index:
            if res_gens.loc[gen].p_nom_extendable:
                lhs += linexpr((cc_ds.loc[gen], get_var(net, 'Generator', 'p_nom')[gen]))
            else:
                legacy_at_bus += cc_ds.loc[gen] * res_gens.loc[gen].p_nom_min

        # Get load for country
        load_idx = net.loads[net.loads.bus == bus].index
        load_peak = net.loads_t.p_set[load_idx].max()

        load_corrected_with_margin = load_peak * (1 + prm)
        rhs = load_corrected_with_margin.values[0] - legacy_at_bus

        define_constraints(net, lhs.sum(), '>=', rhs, 'planning_reserve_margin', bus)
