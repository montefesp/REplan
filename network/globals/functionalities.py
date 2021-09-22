import pandas as pd

import pypsa

from iepy.technologies import get_config_values
from iepy.geographics import get_subregions

import logging
logger = logging.getLogger()


def add_extra_functionalities(net: pypsa.Network, snapshots: pd.DatetimeIndex):
    """
    Wrapper for the inclusion of multiple extra_functionalities.

    Parameters
    ----------
    net: pypsa.Network
        A PyPSA Network instance with buses associated to regions
        and containing a functionality configuration dictionary
    snapshots: pd.DatetimeIndex
        Network snapshots.

    """

    assert hasattr(net, 'config'), 'To use functionalities, you need to give the network a config attribute' \
                                   'specifying which functionality you want to add.'

    mandatory_fields = ['functionalities', 'pyomo']
    for field in mandatory_fields:
        assert field in net.config, f'Error: No field {field} found in config.'
    conf_func = net.config["functionalities"]

    pyomo = net.config['pyomo']
    if pyomo:
        import network.globals.pyomo as funcs
    else:
        import network.globals.nomopyomo as funcs

    # Some functionalities are currently only implemented in pyomo
    if 'snsp' in conf_func and conf_func["snsp"]["include"]:
        if pyomo:
            funcs.add_snsp_constraint(net, snapshots, conf_func["snsp"]["share"])
        else:
            logger.warning('SNSP functionality is currently not implented in nomopyomo')

    if 'curtailement' in conf_func and conf_func["curtailment"]["include"]:
        if pyomo:
            strategy = conf_func["curtailment"]["strategy"][0]
            if strategy == 'economic':
                funcs.add_curtailment_penalty_term(net, snapshots, conf_func["curtailment"]["strategy"][1])
            elif strategy == 'technical':
                funcs.add_curtailment_constraints(net, snapshots, conf_func["curtailment"]["strategy"][1])
        else:
            logger.warning('Curtailement functionality is currently not implented in nomopyomo')

    if "co2_emissions" in conf_func and conf_func["co2_emissions"]["include"]:
        strategy = conf_func["co2_emissions"]["strategy"]
        mitigation_factor = conf_func["co2_emissions"]["mitigation_factor"]
        ref_year = conf_func["co2_emissions"]["reference_year"]
        if strategy == 'country':
            # TODO: this is not very robust
            countries = get_subregions(net.config['region'])
            assert len(countries) == len(mitigation_factor), \
                "Error: a co2 emission reduction share must be given for each country in the main region."
            mitigation_factor_dict = dict(zip(countries, mitigation_factor))
            funcs.add_co2_budget_per_country(net, mitigation_factor_dict, ref_year)
        elif strategy == 'global':
            funcs.add_co2_budget_global(net, net.config["region"], mitigation_factor, ref_year)

    if 'import_limit' in conf_func and conf_func["import_limit"]["include"]:
        funcs.add_import_limit_constraint(net, conf_func["import_limit"]["share"])
        if pyomo:
            # TODO: this is not very robust
            countries = get_subregions(net.config['region'])
            funcs.add_import_limit_constraint(net, conf_func["import_limit"]["share"], countries)

    if 'techs' in net.config and 'battery' in net.config["techs"] and not net.config["techs"]["battery"]["fixed_duration"] and net.stores['e_nom_extendable'].all():
        ctd_ratio = get_config_values("Li-ion_p", ["ctd_ratio"])
        funcs.store_links_constraint(net, ctd_ratio)

    if "disp_cap" in conf_func and conf_func["disp_cap"]["include"]:
        countries = get_subregions(net.config['region'])
        disp_threshold = conf_func["disp_cap"]["disp_threshold"]
        assert len(countries) == len(disp_threshold), \
            "A dispatchable capacity threshold must be given for each country in the main region."
        thresholds = dict(zip(countries, disp_threshold))
        funcs.dispatchable_capacity_lower_bound(net, thresholds)

    if 'prm' in conf_func and conf_func["prm"]["include"]:
        prm = conf_func["prm"]["PRM"]
        funcs.add_planning_reserve_constraint(net, prm)

    if 'mga' in conf_func and conf_func['mga']['include']:
        if not pyomo:
            epsilon = conf_func['mga']['epsilon']
            funcs.min_links_capacity(net, epsilon)
        else:
            logger.warning('MGA functionality is currently not implented in nomopyomo')
