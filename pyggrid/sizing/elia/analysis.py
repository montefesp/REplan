from os.path import join, dirname, abspath

import matplotlib.pyplot as plt

from shapely.ops import unary_union

from pyggrid.postprocessing.results_display import *
from pyggrid.postprocessing.plotly import SizingPlotly
from pyggrid.data.topologies.core.plot import plot_topology
from pyggrid.postprocessing.utils import *
from pyggrid.data.geographics import get_shapes
from pyggrid.data.geographics.plot import display_polygons


if __name__ == '__main__':

    from pypsa import Network

    year = 2016

    techs = ["ccgt", "pv_residential_national", "pv_utility_national",
             "wind_offshore_national", "wind_onshore_national",
             "AC", "DC", "Li-ion"]
    techs_generic = ["ccgt", "pv_residential", "pv_utility",
                     "wind_offshore", "wind_onshore",
                     "AC", "DC", "Li-ion"]
    cases = ["EU", "EUNA", "EUGLIS", "EUNAGLIS"]

    capacities_df = pd.DataFrame(columns=cases, index=techs_generic)
    for case in cases:
        output_dir = join(dirname(abspath(__file__)), f'../../../output/from_pan/{case}_{year}/')
        net = Network()
        net.import_from_csv_folder(output_dir)

        #sp = SizingPlotly(output_dir)
        #fig = sp.show_topology_heatmap()
        #fig.write_html(f"{output_dir_na}topology_heatmap.html", auto_open=True)
        #exit()

        # display_generation(net)
        # display_transmission(net)
        # display_storage(net)
        # display_co2(net)

        # Compare objectives (i.e. total cost of the system)
        cost = get_generators_cost(net).sum() + get_links_capex(net).sum() + get_storage_cost(net).sum()
        print(f"Generator cost {get_generators_cost(net).sum():.3f} (B€)")
        print(f"Link cost {get_links_capex(net).sum():.3f} (B€)")
        print(f"Storage cost {get_storage_cost(net).sum():.3f} (B€)")

        # Compare total capacities installed
        capacities = pd.concat([get_generators_capacity(net)["new"],
                                get_links_capacity(net)["new [TWkm]"],
                                get_storage_power_capacity(net)["new [GW]"]])
        capacities_df[case] = capacities[techs].round().values

        # TODO: what if we limited potential in NA? Because for now, considering the full territory is accessible
        # and not adding connection cost is not very representative
        # TODO: actually we have kind of the same problem with northen countries...
        print(net.generators[net.generators.type == "pv_utility_noneu"][["p_nom_new", "p_nom_max"]].round())
        print(net.generators[net.generators.type == "wind_onshore_noneu"][["p_nom_new", "p_nom_max"]].round())


        # TODO: could try to test a case where we increase greenland capacity potential?
        # --> it is super interesting to see that GL is actually use even if the average performance of the signal
        # are less good than for IS --> complementarity appearing?
        print(net.generators_t.p_max_pu[net.generators[net.generators.type == "wind_onshore_noneu"].index].mean())
        print(net.generators_t.p_max_pu[net.generators[net.generators.type == "wind_onshore_noneu"].index].median())

        # --> We clearly see that LY capacity factory are worse than any others
        if 0:
            countries = ["DZ", "EG", "MA", "LY", "TN"]
            quantiles = np.linspace(0, 1, 11)
            cap_factor_in_na_countries = pd.DataFrame(columns=quantiles, index=countries)
            for c in countries:
                cap_factor_in_na_countries.loc[c] = get_generators_cap_factors(net, [c], ["pv_utility_noneu"]).loc["pv_utility_noneu"]
            print(cap_factor_in_na_countries)
            ax = cap_factor_in_na_countries.T.plot()
            plt.show()


    capacities_df.plot(kind="bar", title=f"Year {year}")
    plt.xticks(rotation="45")
    plt.savefig(f"capacities_{year}.png", bbox_inches="tight")

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

    # Plot map of regions
    if 0:
        region_shapes = []
        for region in regions_dict.keys():
            region_shape = unary_union(get_shapes(regions_dict[region], "onshore")["geometry"].values)
            region_shapes += [region_shape]
        print(region_shapes)
        display_polygons(region_shapes, show=True)

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
    ax.set_ylim([0, 275])
    ax.grid()

    capacities_per_region_na_new = pd.DataFrame(index=techs, columns=regions_dict.keys())
    for region in regions_dict.keys():
        cap_df = get_generators_capacity(net_na, regions_dict[region], techs)["init"]
        capacities_per_region_na_new.loc[cap_df.index, region] = cap_df
    ax = capacities_per_region_na_new.T.plot(kind="bar")
    capacities_per_region_na_tot.T.plot(ax=ax, kind="bar", alpha=0.5)
    ax.set_ylim([0, 275])
    ax.grid()

    plt.show()

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
