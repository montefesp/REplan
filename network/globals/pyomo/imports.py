from typing import List

from pyomo.environ import Constraint, NonNegativeReals
import pypsa

from epippy.load import get_load


def add_import_limit_constraint(network: pypsa.Network, import_share: float, countries: List[str]):
    """
    Add per-bus constraint on import budgets.

    Parameters
    ----------
    network: pypsa.Network
        A PyPSA Network instance with buses associated to regions
    import_share: float
        Maximum share of load that can be satisfied via imports.
    countries: List[str]
        ISO2 codes of countries on which to impose import limit constraints

    Notes
    -----
    Using a flat value across EU, could be updated to support different values for different countries.

    WARNING: this function works based on the assumption that the bus is associated to a country. Should be updated.

    """

    model = network.model
    links = network.links
    snapshots = network.snapshots

    def import_constraint_rule(model, bus):

        load_at_bus = get_load(timestamps=snapshots, countries=[bus], missing_data='interpolate').sum()
        import_budget = import_share * load_at_bus.values[0]

        links_in = links[links.bus1 == bus].index
        links_out = links[links.bus0 == bus].index

        imports = 0.
        if not links_in.empty:
            imports += sum(model.link_p[e, s] for e in links_in for s in network.snapshots)
        if not links_out.empty:
            imports -= sum(model.link_p[e, s] for e in links_out for s in network.snapshots)
        return imports <= import_budget

    model.import_constraint = Constraint(countries, rule=import_constraint_rule)
