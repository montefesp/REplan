from typing import Dict

from pyomo.environ import Constraint, NonNegativeReals
import pypsa


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
    model = net.model
    buses = net.loads.bus
    dispatchable_technologies = ['ocgt', 'ccgt', 'ccgt_ccs', 'nuclear', 'sto']

    def dispatchable_capacity_constraint_rule(model, bus):

        if bus in thresholds.keys():

            lhs = 0
            legacy_at_bus = 0

            gens = net.generators[(net.generators.bus == bus) & (net.generators.type.isin(dispatchable_technologies))]
            for gen in gens.index:
                if gens.loc[gen].p_nom_extendable:
                    lhs += model.generator_p_nom[gen]
                else:
                    legacy_at_bus += gens.loc[gen].p_nom_min

            stos = net.storage_units[(net.storage_units.bus == bus) &
                                     (net.storage_units.type.isin(dispatchable_technologies))]
            for sto in stos.index:
                if stos.loc[sto].p_nom_extendable:
                    lhs += model.storage_unit_p_nom[gen]
                else:
                    legacy_at_bus += stos.loc[sto].p_nom_min

            # Get load for country
            load_idx = net.loads[net.loads.bus == bus].index
            load_peak = net.loads_t.p_set[load_idx].max()

            load_peak_threshold = load_peak * thresholds[bus]
            rhs = max(0, load_peak_threshold.values[0] - legacy_at_bus)

            return lhs >= rhs

    model.dispatchable_capacity_constraint = Constraint(buses, rule=dispatchable_capacity_constraint_rule)


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
    model = net.model
    buses = net.loads.bus
    cc_ds = net.cc_ds
    dispatchable_technologies = ['ocgt', 'ccgt', 'ccgt_ccs', 'nuclear', 'sto']
    res_technologies = ['wind_onshore', 'wind_offshore', 'pv_utility', 'pv_residential']

    def planning_reserve_constraint_rule(model, bus):

        lhs = 0
        legacy_at_bus = 0

        gens = net.generators[(net.generators.bus == bus) & (net.generators.type.isin(dispatchable_technologies))]
        for gen in gens.index:
            if gens.loc[gen].p_nom_extendable:
                lhs += model.generator_p_nom[gen]
            else:
                legacy_at_bus += gens.loc[gen].p_nom_min

        stos = net.storage_units[(net.storage_units.bus == bus) &
                                 (net.storage_units.type.isin(dispatchable_technologies))]
        for sto in stos.index:
            if stos.loc[sto].p_nom_extendable:
                lhs += model.storage_unit_p_nom[gen]
            else:
                legacy_at_bus += stos.loc[sto].p_nom_min

        res_gens = net.generators[(net.generators.bus == bus) &
                                  (net.generators.type.str.contains('|'.join(res_technologies)))]
        for gen in res_gens.index:
            lhs += model.generator_p_nom[gen] * cc_ds.loc[' '.join(gen.split(' ')[1:])]

        # Get load for country
        load_idx = net.loads[net.loads.bus == bus].index
        load_peak = net.loads_t.p_set[load_idx].max()

        load_corrected_with_margin = load_peak * (1 + prm)
        rhs = load_corrected_with_margin.values[0] - legacy_at_bus

        return lhs >= rhs

    model.planning_reserve_margin = Constraint(buses, rule=planning_reserve_constraint_rule)
