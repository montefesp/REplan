from os.path import join, dirname, abspath

import matplotlib.pyplot as plt

from pyggrid.postprocessing.results_display import *
from pyggrid.postprocessing.plotly import SizingPlotly
from pyggrid.data.topologies.core.plot import plot_topology
from pyggrid.postprocessing.utils import *

if __name__ == '__main__':

    from pypsa import Network
    output_dir_na = join(dirname(abspath(__file__)), f'../../../output/from_pan/20200818_094604/')
    net_na = Network()
    net_na.import_from_csv_folder(output_dir_na)
    output_dir_eu = join(dirname(abspath(__file__)), f'../../../output/from_pan/20200818_170748/')
    net_eu = Network()
    net_eu.import_from_csv_folder(output_dir_eu)



    #sp = SizingPlotly(output_dir_full)
    #fig = sp.plot_topology()
    #fig.write_html(f"{output_dir_full}topology.html", auto_open=True)

    # display_generation(net)
    # display_transmission(net)
    # display_storage(net)
    # display_co2(net)

    # Compare objectives (i.e. total cost of the system)
    eu_cost = get_generators_cost(net_eu).sum() + get_links_capex(net_eu).sum() + get_storage_cost(net_eu).sum()
    na_cost = get_generators_cost(net_na).sum() + get_links_capex(net_na).sum() + get_storage_cost(net_na).sum()
    print(f"EU alone: {eu_cost:.3f} vs EU+NA: {na_cost:.3f} (B€)")

    # Compare total capacities installed
    capacities_eu = pd.concat([get_generators_capacity(net_eu)["final"],
                               get_links_capacity(net_eu)["final [GW]"],
                               get_storage_power_capacity(net_eu)["final [GW]"]])
    capacities_na = pd.concat([get_generators_capacity(net_na)["final"],
                               get_links_capacity(net_na)["final [GW]"],
                               get_storage_power_capacity(net_na)["final [GW]"]])
    print(pd.concat([capacities_eu, capacities_na], axis=1, sort=True))

    # See capacity distribution
    regions_dict = {'Iberia': ['ES', 'PT'],
                    'Central West': ['NL', 'BE', 'LU', 'FR', 'DE'],
                    'Nordics': ['DK', 'NO', 'SE', 'FI'],
                    'British Isles': ['GB', 'IE'],
                    'Central South': ['CH', 'IT', 'AT', 'SI'],
                    'East': ['PL', 'HU', 'CZ', 'SK', 'LV', 'LT', 'EE'],
                    'South': ['HR', 'GR', 'AL', 'ME', 'BA', 'RS', 'BG', 'RO']}
    techs = ["ccgt", "pv_residential_national", "pv_utility_national",
             "wind_offshore_national", "wind_onshore_national"]

    # Total

    capacities_per_region_eu_tot = pd.DataFrame(index=techs, columns=regions_dict.keys())
    for region in regions_dict.keys():
        cap_df = get_generators_capacity(net_eu, regions_dict[region], techs)["final"]
        capacities_per_region_eu_tot.loc[cap_df.index, region] = cap_df

    capacities_per_region_na_tot = pd.DataFrame(index=techs, columns=regions_dict.keys())
    for region in regions_dict.keys():
        cap_df = get_generators_capacity(net_na, regions_dict[region], techs)["final"]
        capacities_per_region_na_tot.loc[cap_df.index, region] = cap_df

    # New
    capacities_per_region_eu_new = pd.DataFrame(index=techs, columns=regions_dict.keys())
    for region in regions_dict.keys():
        cap_df = get_generators_capacity(net_eu, regions_dict[region], techs)["init"]
        capacities_per_region_eu_new.loc[cap_df.index, region] = cap_df
    ax = capacities_per_region_eu_new.T.plot(kind="bar")
    capacities_per_region_eu_tot.T.plot(ax=ax, kind="bar", alpha=0.5)

    capacities_per_region_na_new = pd.DataFrame(index=techs, columns=regions_dict.keys())
    for region in regions_dict.keys():
        cap_df = get_generators_capacity(net_na, regions_dict[region], techs)["init"]
        capacities_per_region_na_new.loc[cap_df.index, region] = cap_df
    ax = capacities_per_region_na_new.T.plot(kind="bar")
    capacities_per_region_na_tot.T.plot(ax=ax, kind="bar", alpha=0.5)


    # Distribution of capacity in NA
    # --> very intriguing that some capacity is installed in EG rather than in LY -> solar so good there?
    # --> or very good complementarity?
    countries = ["DZ", "EG", "MA", "LY", "TN"]
    capacities_in_na_countries = pd.Series(index=countries)
    capacities_in_na_countries_max = pd.Series(index=countries)
    for c in countries:
        capacities_in_na_countries[c] = get_generators_capacity(net_na, [c], ["pv_utility_noneu"])["new"]
        capacities_in_na_countries_max[c] = get_generators_capacity(net_na, [c], ["pv_utility_noneu"])["max"]
    plt.figure()
    ax = capacities_in_na_countries.plot(kind="bar")
    # capacities_in_na_countries_max.plot(ax=ax, kind="bar", alpha=0.5)

    # --> We clearly see that LY capacity factory are worse than any others
    quantiles = np.linspace(0, 1, 11)
    cap_factor_in_na_countries = pd.DataFrame(columns=quantiles, index=countries)
    for c in countries:
        cap_factor_in_na_countries.loc[c] = get_generators_cap_factors(net_na, [c], ["pv_utility_noneu"]).loc["pv_utility_noneu"]
    print(cap_factor_in_na_countries)
    plt.figure()
    ax = cap_factor_in_na_countries.T.plot()

    # --> Question: could it be that capacity is installed at bus only where the higher quantiles are the highest?
    countries = ["ES", "PT", "GR", "IT"]
    cap_factor_in_eu_countries = pd.DataFrame(columns=quantiles, index=countries)
    for c in countries:
        cap_factor_in_eu_countries.loc[c] = get_generators_cap_factors(net_eu, [c], ["pv_utility_national"]).loc["pv_utility_national"]
    print(cap_factor_in_eu_countries)
    cap_factor_in_eu_countries.T.plot(ax=ax, alpha=0.5)
    plt.show()


    # Compare storage capacities --> would be interesting to compare to PV installed
    capacities_per_region = pd.DataFrame(index=["EU", "NA"], columns=regions_dict.keys())
    for region in regions_dict.keys():
        capacities_per_region.loc["EU", region] = \
            get_storage_power_capacity(net_eu, regions_dict[region], ["Li-ion"]).loc["Li-ion", "new [GW]"]
        capacities_per_region.loc["NA", region] = \
            get_storage_power_capacity(net_na, regions_dict[region], ["Li-ion"]).loc["Li-ion", "new [GW]"]
    ax = capacities_per_region.plot(kind="bar")

    # TODO: how to do the same thing for transmission?

    # See where we reach maximum capacity
    # --> very far away from max capacity in each region (except maybe wind onshore in central west >more than half)
    capacities_per_region_eu = pd.DataFrame(index=techs, columns=regions_dict.keys())
    capacities_per_region_eu_max = pd.DataFrame(index=techs, columns=regions_dict.keys())
    for region in regions_dict.keys():
        cap_df = get_generators_capacity(net_eu, regions_dict[region], techs)
        capacities_per_region_eu.loc[cap_df.index, region] = cap_df["final"]
        capacities_per_region_eu_max.loc[cap_df.index, region] = cap_df["max"]
    ax = capacities_per_region_eu.T.plot(kind="bar")
    ax = capacities_per_region_eu_max.T.plot(ax=ax, kind="bar", alpha=0.5)

    plt.show()


    # TODO: need to generate results with duals if I want to have marginal prices
    # marginal_price = pypsa.linopt.get_dual(net_, 'Bus', 'marginal_price')
    # shadow_price = pypsa.linopt.get_dual(net_, 'Generator', 'mu_upper')
    # print((shadow_price < 0).sum())
    # print((pypsa.linopt.get_dual(net_, 'Generator', 'mu_lower') < 0).sum())
    # print(net_.dualvalues)
