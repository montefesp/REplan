import pypsa
import pandas as pd
from pyomo.environ import Constraint, Var, NonNegativeReals
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

    snsp_share = config['functionalities']["snsp"]["share"]

    model = network.model

    nonsync_techs = ['wind', 'pv', 'Li-ion']

    def snsp_rule(model, bus, snapshot):

        generators_at_bus = network.generators.index[network.generators.bus == bus]
        generation_at_bus_nonsync = generators_at_bus[generators_at_bus.str.contains('|'.join(nonsync_techs))]

        storage_at_bus = network.storage_units.index[network.storage_units.bus == bus]
        storage_at_bus_nonsync = storage_at_bus[storage_at_bus.str.contains('|'.join(nonsync_techs))]

        return (sum(model.generator_p[gen,snapshot] for gen in generation_at_bus_nonsync) +
                sum(model.storage_p_dispatch[gen,snapshot] for gen in storage_at_bus_nonsync)) <= \
                snsp_share * (sum(model.generator_p[gen, snapshot] for gen in generators_at_bus) +
                                              sum(model.storage_p_dispatch[gen, snapshot] for gen in storage_at_bus))

    model.snsp = Constraint(list(network.buses.index), list(snapshots), rule=snsp_rule)


def add_curtailment_penalty_term(network: pypsa.Network, snapshots: pd.DatetimeIndex):
    """
    Add curtailment penalties to the objective function.

    Parameters
    ----------
    network: pypsa.Network
        A PyPSA Network instance with buses associated to regions
    snapshots: pd.DatetimeIndex
        Network snapshots.

    """

    config_fn = join(dirname(abspath(__file__)), '../sizing/tyndp2018/config.yaml')
    config = yaml.load(open(config_fn, 'r'), Loader=yaml.FullLoader)

    curtailment_cost = config["functionalities"]["curtailment"]["strategy"][1]

    techs = ['wind', 'pv']
    gens = network.generators.index[network.generators.index.str.contains('|'.join(techs))]

    model = network.model
    gens_p_max_pu = network.generators_t.p_max_pu

    model.generator_c = Var(gens, snapshots, within=NonNegativeReals)

    def generation_curtailment_rule(model, gen, snapshot):
        return model.generator_c[gen, snapshot] == \
               model.generator_p_nom[gen] * gens_p_max_pu.loc[snapshot, gen] - model.generator_p[gen, snapshot]
    model.generation_curtailment = Constraint(list(gens), list(snapshots), rule=generation_curtailment_rule)

    model.objective.expr += curtailment_cost * sum(model.generator_c[gen, s] for gen in gens for s in snapshots)


def add_curtailment_constraints(network: pypsa.Network, snapshots: pd.DatetimeIndex):
    """
    Add extra constrains limiting curtailment of each generator, at each time step, as a share of p_max_pu*p_nom.

    Parameters
    ----------
    network: pypsa.Network
        A PyPSA Network instance with buses associated to regions
    snapshots: pd.DatetimeIndex
        Network snapshots.

    """

    config_fn = join(dirname(abspath(__file__)), '../sizing/tyndp2018/config.yaml')
    config = yaml.load(open(config_fn, 'r'), Loader=yaml.FullLoader)

    allowed_curtailment_share = config["functionalities"]["curtailment"]["strategy"][1]

    model = network.model
    gens_p_max_pu = network.generators_t.p_max_pu

    techs = ['wind', 'pv']
    gens = network.generators.index[network.generators.index.str.contains('|'.join(techs))]

    model.generator_c = Var(gens, snapshots, within=NonNegativeReals)

    def generation_curtailment_rule(model, gen, snapshot):
        return model.generator_c[gen, snapshot] == \
               model.generator_p_nom[gen] * gens_p_max_pu.loc[snapshot, gen] - model.generator_p[gen, snapshot]
    model.generation_curtailment = Constraint(list(gens), list(snapshots), rule=generation_curtailment_rule)

    def curtailment_rule(model, gen, snapshot):
        return model.generator_c[gen, snapshot] <= \
               allowed_curtailment_share * gens_p_max_pu.loc[snapshot, gen] * model.generator_p_nom[gen]
    model.limit_curtailment = Constraint(list(gens), list(snapshots), rule=curtailment_rule)


def add_extra_functionalities(network: pypsa.Network, snapshots: pd.DatetimeIndex):
    """
    Wrapper for the inclusion of multiple extra_functionalities.

    Parameters
    ----------
    network: pypsa.Network
        A PyPSA Network instance with buses associated to regions
    snapshots: pd.DatetimeIndex
        Network snapshots.

    """

    config_fn = join(dirname(abspath(__file__)), '../sizing/tyndp2018/config.yaml')
    config = yaml.load(open(config_fn, 'r'), Loader=yaml.FullLoader)

    if config["functionalities"]["snsp"]["include"]:
        add_snsp_constraint_tyndp(network, snapshots)
    if config["functionalities"]["curtailment"]["include"]:
        strategy = config["functionalities"]["curtailment"]["strategy"][0]
        if strategy == 'economic':
            add_curtailment_penalty_term(network, snapshots)
        elif strategy == 'technical':
            add_curtailment_constraints(network, snapshots)
