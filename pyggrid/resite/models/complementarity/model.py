from typing import Dict

from pyggrid.resite.models.complementarity.utils import *


def build_model(resite, modelling, params: Dict):

    # TODO: this is the same as resource_quality_mapping
    delta = 4
    measure = "mean"
    cap_factor_per_window_df = resource_quality_mapping(resite.data_dict["cap_factor_df"], delta, measure)

    # TODO: computing the criticality window matrix
    alpha = "load_central"  # "load_partition"
    norm_type = "min"
    # retrieve_load_data_partitions("/home/utilisateur/Global_Grid/code/pyggrid/data/load/generated", 0, 0, 0, 0, 0)
    critical_window_mapping(cap_factor_per_window_df, alpha, delta, ["BENELUX"], resite.data_dict["load"], norm_type)
