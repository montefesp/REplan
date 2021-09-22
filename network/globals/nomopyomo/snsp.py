import pypsa
from pypsa.linopt import get_var, linexpr, define_constraints


def add_snsp_constraint(net: pypsa.Network, snsp_share: float):
    """
    Add system non-synchronous generation share constraint to the model.

    Parameters
    ----------
    net: pypsa.Network
        A PyPSA Network instance with buses associated to regions
    snsp_share: float
        Share of system non-synchronous generation.

    """
    snapshots = net.snapshots

    nonsync_gen_types = 'wind|pv'
    gens_p = get_var(net, 'Generator', 'p')
    load_p = net.loads_t.p_set

    for s in snapshots:

        gens_p_s = gens_p.loc[s, :]
        nonsync_gen_ids = net.generators.index[(net.generators.type.str.contains(nonsync_gen_types))]
        nonsyncgen_p = gens_p_s.loc[nonsync_gen_ids]
        lhs_gen_nonsync = linexpr((1., nonsyncgen_p)).values

        load_p_s = load_p.loc[s, :].sum()
        rhs = linexpr((-snsp_share, load_p_s)).values

        define_constraints(net, lhs_gen_nonsync, '<=', rhs, 'snsp_constraint', s)
