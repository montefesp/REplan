from pyggrid.data.technologies.costs import *


def test_get_costs_output_format():
    techs = ["ccgt", "ocgt", "nuclear", "sto", "ror", "phs", "wind_onshore", "wind_offshore", "wind_floating",
             "pv_utility", "pv_residential", "Li-ion", "AC", "DC"]
    for tech in techs:
        assert isinstance(get_costs(tech, 1), tuple)


def test_compute_capital_cost():
    # Example values for CCGT
    fom = 26  # M€/GW*y
    capex = 900  # M€/GW
    lifetime = 30  # years
    assert compute_capital_cost(fom, capex, lifetime) == 56


def test_compute_marginal_cost():
    # Example for a cable
    assert compute_marginal_cost(0) == 0.
    # Example for a wind turbine
    assert compute_marginal_cost(0.0012) == 0.0012  # M€/GWh
    # Example for CCGT
    vom = 0.004  # M€/GWhel
    fuel_cost = 0.03  # for gas, M€/GWhth
    efficiency = 0.55
    co2_content = 0.225  # for gas, kT/GWhth
    co2_cost = 0.04  # M€/kT
    m_cost = compute_marginal_cost(vom, fuel_cost, efficiency, co2_content, co2_cost)
    assert round(m_cost, 4) == 0.0749
