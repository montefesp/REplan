from postprocessing.utils import *
from iepy.technologies import get_costs


def display_generation(net: pypsa.Network):
    """Display information about generation"""

    print('# --- GENERATION --- #')

    cap_cost, marg_cost = get_gen_capital_and_marginal_cost(net)
    costs = pd.concat(
        [cap_cost.rename('ccost'), marg_cost.rename('mcost'), get_generators_capex(net), get_generators_opex(net)],
        axis=1)
    costs = costs.round(4)
    print(f"Costs (M$):\n{costs}\n")

    capacities = get_generators_capacity(net)
    print(f"Generators capacity (GW):\n{capacities}\n")

    cf_df = get_generators_average_usage(net).rename('CF [%]')
    curt_df = get_generators_curtailment(net).rename('Curt.')
    curt_cf_df = pd.concat([cf_df, curt_df], axis=1)
    gen_df = get_generators_generation(net).rename('Gen.').to_frame()
    df_gen = pd.concat([gen_df, curt_cf_df], axis=1, sort=False)
    print(f"Generators generation (TWh):\n{df_gen}\n")

    print(f"Total generation (TWh):\n{get_generators_generation(net).sum()}\n")
    print(f"Total load (TWh):\n{round(net.loads_t.p.values.sum() * 1e-3, 2)}\n")


def display_transmission(net: pypsa.Network):
    """Display information about transmission"""

    print('\n\n\n# --- TRANSMISSION --- #')

    if len(net.links.index) != 0:
        links_capacities = get_links_capacity(net)
        print(f"Links capacity:\n{links_capacities}\n")

        df_power = get_links_power(net)
        df_cf = get_links_usage(net)
        df_capex = get_links_capex(net)

        df_links = pd.concat([df_power.rename('Flows [TWh]'),
                              df_cf.rename('CF [%]'),
                              df_capex.rename('capex [M$]')], axis=1)
        df_links.loc['AC', 'ccost [M€/GW/km]'] = get_costs('AC', sum(net.snapshot_weightings['objective']))[0]
        df_links.loc['DC', 'ccost [M€/GW/km]'] = get_costs('DC', sum(net.snapshot_weightings['objective']))[0]

        print(f"Links flows & costs:\n{df_links}\n")


def display_storage(net: pypsa.Network):
    """Display information about storage"""

    print('\n\n\n# --- STORAGE --- #')

    cap_cost, marg_cost = get_storage_capital_and_marginal_cost(net)
    costs = pd.concat([cap_cost.rename('ccost'), marg_cost.rename('mcost'), get_storage_capex(net),
                       get_storage_opex(net)], axis=1)
    costs = costs.round(4)
    print(f"Costs (M$):\n{costs}\n")

    capacities_p = get_storage_power_capacity(net)
    capacities_e = get_storage_energy_capacity(net)
    capacities = pd.concat([capacities_p, capacities_e], axis=1)
    print(f"Storage capacities:\n{capacities}\n")

    storage_f = get_storage_energy_in(net)
    storage_e = capacities_e['init [GWh]'] + capacities_e['new [GWh]']
    cycles = (storage_f / storage_e).round(0)
    print(f"Storage cycles:\n{cycles}\n")

    spillage = get_storage_spillage(net)
    print(f"Storage spillage [GWh]:\n{spillage.round(2)}\n")


def display_co2(net: pypsa.Network):
    print('\n\n\n# --- CO2 --- #')

    df_co2 = pd.DataFrame(index=['CO2'], columns=['budget [Mt]', 'use [Mt]', 'use [%]'])

    generators_t = net.generators_t['p']
    generators_t_year = generators_t.sum(axis=0)

    generators_specs = net.generators[['carrier', 'efficiency']].copy()
    generators_emissions = net.carriers
    generators_specs['emissions'] = generators_specs['carrier'].map(generators_emissions['co2_emissions']).fillna(0.)
    generators_specs['emissions_eff'] = generators_specs['emissions'] / generators_specs['efficiency']

    co2_emissions_per_gen = generators_t_year * generators_specs['emissions_eff'] * 1e-3
    co2_emissions = co2_emissions_per_gen.sum()

    # TODO: this generate an error if there is a no co2 budget
    co2_budget = net.global_constraints.constant.values[0] * 1e-3

    df_co2.loc['CO2', 'budget [Mt]'] = co2_budget
    df_co2.loc['CO2', 'use [Mt]'] = co2_emissions
    df_co2.loc['CO2', 'use [%]'] = co2_emissions / co2_budget

    df_co2 = df_co2.round(2)

    print(f"CO2 utilization:\n{df_co2}\n")
