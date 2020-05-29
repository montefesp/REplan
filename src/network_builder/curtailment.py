import pypsa
import pandas as pd
from os.path import join, abspath, dirname
import yaml

def add_curtailment_penalty_term(network: pypsa.Network, snapshots: pd.DatetimeIndex):
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

    curtailment_cost = config['functionality']['curtailment']['cost']

    techs = ['wind', 'pv']
    gens = network.generators.index[network.generators.index.str.contains('|'.join(techs))]

    model = network.model

    model.objective.expr += sum(curtailment_cost *
                                (model.generator_p_nom[gen] * network.generators_t.p_max_pu.loc[snapshot, gen]
                                - model.generator_p[gen, snapshot]) for gen in gens for snapshot in snapshots)
