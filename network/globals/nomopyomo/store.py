import pypsa
from pypsa.linopt import get_var, linexpr, define_constraints


def store_links_constraint(net: pypsa.Network, ctd_ratio: float):
    """
    Constraint that links the charging and discharging ratings of store units.

    Parameters
    ----------
    net: pypsa.Network
        A PyPSA Network instance with buses associated to regions
        and containing a functionality configuration dictionary
    ctd_ratio: float
        Pre-defined charge-to-discharge ratio for such units.

    """

    links_p_nom = get_var(net, 'Link', 'p_nom')

    links_to_bus = links_p_nom[links_p_nom.index.str.contains('to AC')].index
    links_from_bus = links_p_nom[links_p_nom.index.str.contains('AC to')].index

    for pair in list(zip(links_to_bus, links_from_bus)):

        discharge_link = links_p_nom.loc[pair[0]]
        charge_link = links_p_nom.loc[pair[1]]
        lhs = linexpr((ctd_ratio, discharge_link), (-1., charge_link))

        define_constraints(net, lhs, '==', 0., 'store_links_constraint')
