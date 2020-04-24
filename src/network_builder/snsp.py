import pypsa
from pypsa.opt import LExpression
import pandas as pd
from pyomo.environ import Constraint
from os.path import join, abspath, dirname
import yaml


def add_snsp_constraint_tyndp(network: pypsa.Network, snapshots: pd.DatetimeIndex):
    """
    Add system non-synchronous generation share constraint to the model.

    Parameters
    ----------
    network: pypsa.Network
        A PyPSA Network instance with buses associated to regions
    snapshots: pd.DatetimeIndex
        Network snapshots.

    """

    config_fn = join(dirname(abspath(__file__)), '../sizing/tyndp2018/config.yaml')
    config = yaml.load(open(config_fn, 'r'), Loader=yaml.FullLoader)

    model = network.model

    nonsync_techs = ['wind', 'pv', 'Li-ion']

    def snsp_rule(model,bus,snapshot):

        generators_at_bus = network.generators.index[network.generators.bus == bus]
        generation_at_bus_nonsync = generators_at_bus[generators_at_bus.str.contains('|'.join(nonsync_techs))]

        storage_at_bus = network.storage_units.index[network.storage_units.bus == bus]
        storage_at_bus_nonsync = storage_at_bus[storage_at_bus.str.contains('|'.join(nonsync_techs))]

        return (sum(model.generator_p[gen,snapshot] for gen in generation_at_bus_nonsync)+ \
               sum(model.storage_p_dispatch[gen,snapshot] for gen in storage_at_bus_nonsync))  \
                <= config["snsp"]["share"] * (sum(model.generator_p[gen, snapshot] for gen in generators_at_bus) +
                                              sum(model.storage_p_dispatch[gen, snapshot] for gen in storage_at_bus))

    model.snsp = Constraint(list(network.buses.index), list(snapshots), rule=snsp_rule)