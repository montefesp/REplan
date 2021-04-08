import pandas as pd

from pyomo.environ import Constraint, NonNegativeReals
import pypsa


def add_snsp_constraint_tyndp(net: pypsa.Network, snapshots: pd.DatetimeIndex, snsp_share: float):
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
    nonsync_gen_types = 'wind|pv'
    nonsync_gen_ids = net.generators.index[net.generators.type.str.contains(nonsync_gen_types)]
    nonsync_storage_ids = net.storage_units.index[net.storage_units.type == "Li-ion"]

    # Impose for each time step the non-synchronous production be lower than a part of the total production
    def snsp_rule(model, snapshot):

        # Non-synchronous 'production'
        nonsync_gen_p = sum(model.generator_p[idx, snapshot] for idx in nonsync_gen_ids)\
            if len(nonsync_gen_ids) != 0 else 0
        nonsync_storage_dispatch = sum(model.storage_p_dispatch[idx, snapshot] for idx in nonsync_storage_ids)\
            if len(nonsync_storage_ids) != 0 else 0

        # Synchronous production
        full_gen_p = sum(model.generator_p[:, snapshot]) if len(net.generators) != 0 else 0
        full_storage_dispatch = sum(model.storage_p_dispatch[:, snapshot]) if len(net.storage_units) != 0 else 0

        return nonsync_gen_p + nonsync_storage_dispatch <= snsp_share * (full_gen_p + full_storage_dispatch)

    model.snsp = Constraint(list(snapshots), rule=snsp_rule)