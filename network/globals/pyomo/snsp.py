from pyomo.environ import Constraint, NonNegativeReals
import pypsa
import pandas as pd


def add_snsp_constraint(net: pypsa.Network, snapshots: pd.DatetimeIndex, snsp_share: float):
    """
    Add system non-synchronous generation share constraint to the model.

    Parameters
    ----------
    net: pypsa.Network
        A PyPSA Network instance with buses associated to regions
    snapshots: pd.DatetimeIndex
        Network snapshots.
    snsp_share: float
        Share of system non-synchronous generation.

    """

    model = net.model
    load = net.loads_t.p_set
    nonsync_gen_types = 'wind|pv'
    nonsync_gen_ids = net.generators.index[net.generators.type.str.contains(nonsync_gen_types)]

    # Impose for each time step the non-synchronous production be lower than a part of the total production
    def snsp_rule(model, snapshot):

        # Non-synchronous 'production'
        nonsync_gen_p = sum(model.generator_p[idx, snapshot] for idx in nonsync_gen_ids)
        load_at_snapshot = load.loc[snapshot, :].sum()

        return nonsync_gen_p <= snsp_share * load_at_snapshot

    model.snsp = Constraint(list(snapshots), rule=snsp_rule)
