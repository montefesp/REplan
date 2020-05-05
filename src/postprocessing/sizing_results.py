import sys
import os

import pandas as pd
import numpy as np

from pypsa import Network

from src.data.technologies.costs import get_cost


class SizingResults:

    """
    This class allows to extract results from a optimized PyPSA network.
    """

    def __init__(self, network: Network):

        self.net = network

    # --- Generation --- #

    def display_generation(self):
        """Display information about generation"""

        print('# --- GENERATION --- #')

        cap_cost, marg_cost = self.get_gen_capital_and_marginal_cost()
        costs = pd.concat([cap_cost.rename('ccost'), marg_cost.rename('mcost'), self.get_generators_capex(), self.get_generators_opex()], axis=1)
        costs = costs.round(4)
        print(f"Costs (M$):\n{costs}\n")

        capacities = self.get_generators_capacity()
        print(f"Generators capacity (GW):\n{capacities}\n")

        cf_df = self.get_generators_average_usage().rename('CF [%]')
        curt_df = self.get_generators_curtailment().rename('Curt.')
        curt_cf_df = pd.concat([cf_df, curt_df], axis=1)
        gen_df = self.get_generators_generation().rename('Gen.').to_frame()
        df_gen = pd.concat([gen_df, curt_cf_df], axis=1, sort=False)
        print(f"Generators generation (TWh):\n{df_gen}\n")

        print(f"Total generation (TWh):\n{self.get_generators_generation().sum()}\n")
        print(f"Total load (TWh):\n{round(self.net.loads_t.p.values.sum() * (1e-3),2)}\n")
        # print(f"Number of generators:\n{self.get_generators_numbers()}\n")
        # print(f"Generators CFs (%):\n{self.get_generators_average_usage()}\n")
        # print(f"Curtailment (TWh):\n{self.get_generators_curtailment()}\n")

    def get_gen_capital_and_marginal_cost(self):

        gens = self.net.generators.groupby(["type"])
        return gens.capital_cost.mean(), gens.marginal_cost.mean()

    def get_generators_capacity(self):
        """Return the original, new and optimal generation capacities (in MW) for each type of generator."""

        gens = self.net.generators.groupby(["type"])
        init_capacities = gens.p_nom.sum()
        opt_capacities = gens.p_nom_opt.sum()
        max_capacities = gens.p_nom_max.sum()
        new_capacities = opt_capacities - init_capacities

        capacities = pd.concat([init_capacities.rename('init'), new_capacities.rename('new'),
                   opt_capacities.rename('final'), max_capacities.rename('max')], axis=1)

        capacities = capacities.drop(['load']).round(2)

        return capacities

    def get_generators_numbers(self):

        return self.net.generators.groupby("type").count().bus

    def get_generators_generation(self):
        """Return the total generation (in GWh) over the self.net.snapshots for each type of generator."""

        gens = self.net.generators
        types = sorted(list(set(gens.type.values)))
        gens_t = self.net.generators_t

        generation = dict.fromkeys(types)

        for tech_type in types:
            gens_type = gens[gens.type == tech_type]
            generation[tech_type] = gens_t.p[gens_type.index].to_numpy().sum()*(1e-3)

        storage_units_t = self.net.storage_units_t.p
        sto_t = storage_units_t.loc[:,storage_units_t.columns.str.contains("Storage reservoir")]
        generation['sto'] = sto_t.to_numpy().sum()*(1e-3)

        gen_df = pd.DataFrame.from_dict(generation, orient="index", columns=["generation"]).generation

        return gen_df.round(2)

    def get_generators_average_usage(self):
        """Return the average generation capacity usage (i.e. mean(generation_t/capacity)) of each type of generator"""

        opt_cap = self.get_generators_capacity()['final']
        tot_gen = self.get_generators_generation()
        df_capacities_all = self.net.generators
        df_cf_per_generator_all = self.net.generators_t['p_max_pu']

        df_cf = pd.Series(index=opt_cap.index)

        for item in df_cf.index:

            if item in ['wind_onshore', 'wind_offshore', 'wind_floating', 'pv_residential', 'pv_utility']:
                df_capacities = df_capacities_all[df_capacities_all.index.str.contains(item)]['p_nom_opt']
                df_cf_per_generator = df_cf_per_generator_all.loc[:,
                                      df_cf_per_generator_all.columns.str.contains(item)].mean()
                df_cf.loc[item] = np.average(df_cf_per_generator.values, weights=df_capacities.values)

            else:
                df_cf.loc[item] = tot_gen.loc[item]*1e3 / (opt_cap.loc[item]*len(self.net.snapshots))


        return df_cf.round(3)



    def get_generators_curtailment(self):

        opt_cap = self.get_generators_capacity()['final']
        tot_gen = self.get_generators_generation()
        df_capacities_all = self.net.generators
        df_cf_per_generator_all = self.net.generators_t['p_max_pu']

        df_curtailment = pd.Series(index=opt_cap.index)

        for item in df_curtailment.index:

            if item in ['wind_onshore', 'wind_offshore', 'wind_floating', 'pv_residential', 'pv_utility', 'ror']:
                df_capacities = df_capacities_all[df_capacities_all.index.str.contains(item)]['p_nom_opt']
                df_cf_per_generator = df_cf_per_generator_all.loc[:,
                                      df_cf_per_generator_all.columns.str.contains(item)].sum()

                production_per_type = (df_capacities*df_cf_per_generator).sum()
                df_curtailment.loc[item] = production_per_type*(1e-3) - tot_gen.loc[item]

            else:
                df_curtailment.loc[item] = np.nan

        return df_curtailment.round(3)



    def get_generators_opex(self):
        """Return the operational expenses of running each type of generator over the self.net.snapshots"""

        gens = self.net.generators
        types = sorted(list(set(gens.type.values)))
        gens_t = self.net.generators_t

        opex = dict.fromkeys(types)

        for tech_type in types:
            gens_type = gens[gens.type == tech_type]
            generation_per_gen = gens_t.p[gens_type.index]*gens_type.marginal_cost
            opex[tech_type] = generation_per_gen.to_numpy().sum()

        return pd.DataFrame.from_dict(opex, orient="index", columns=["opex"]).opex

    def get_generators_capex(self):
        """Return the capital expenses for building the new capacity for each type of generator."""

        gens = self.net.generators
        gens["p_nom_new"] = gens.p_nom_opt - gens.p_nom
        gens["capex"] = gens.p_nom_new*gens.capital_cost

        return gens.groupby(["type"]).capex.sum()

    def get_generators_cost(self):
        return self.get_generators_opex() + self.get_generators_capex()









    # --- Transmission --- #

    def display_transmission(self):
        """Display information about transmission"""

        print('\n\n\n# --- TRANSMISSION --- #')

        # if len(self.net.lines.index) != 0:
        #     print('LINES:')
        #     lines_capacities = self.get_lines_capacity()
        #     print(f"Lines capacity:\n{lines_capacities}\n")
        #     print(f"Lines power:\n{self.get_lines_power()}\n")
        #     print(f"Lines use:\n{self.get_lines_usage()}\n")
        #     print(f"Lines capex:\n{self.get_lines_capex()}\n")


        if len(self.net.links.index) != 0:
            links_capacities = self.get_links_capacity()
            print(f"Links capacity:\n{links_capacities}\n")

            df_power = self.get_links_power()
            df_cf = self.get_links_usage()
            df_capex = self.get_links_capex()

            df_links = pd.concat([df_power.rename('Flows [TWh]'),
                                  df_cf.rename('CF [%]'),
                                  df_capex.rename('capex [M$]')], axis=1)
            df_links.loc['AC', 'ccost [M€/GW/km]'] = get_cost('AC', len(self.net.snapshots))[0]
            df_links.loc['DC', 'ccost [M€/GW/km]'] = get_cost('DC', len(self.net.snapshots))[0]

            print(f"Links flows & costs:\n{df_links}\n")

            # print(f"Links length:\n{self.get_links_length()}\n")
            # print(f"Links init cap*length:\n{self.get_links_init_cap_length()}\n")
            # print(f"Links new cap*length:\n{self.get_links_new_cap_length()}\n")

    # def get_lines_capacity(self):
    #     """Return the original, new and optimal transmission capacities (in MW) for each type of line."""
    #
    #     lines = self.net.lines.groupby(["carrier"])
    #     init_capacities = lines.s_nom.sum()
    #     opt_capacities = lines.s_nom_opt.sum()
    #     new_capacities = opt_capacities - init_capacities
    #
    #     lines_capacities = pd.concat([init_capacities.rename('init [GW]'),
    #                                   new_capacities.rename('new [GW]'),
    #                                   opt_capacities.rename('final [GW]')], axis=1)
    #
    #     return lines_capacities.round(2)

    def get_links_capacity(self):
        """countries_url_area_types the original, new and optimal transmission capacities (in MW) for links."""

        links = self.net.links[["carrier", "p_nom", "p_nom_opt"]].groupby("carrier").sum()
        links["p_nom_new"] = links["p_nom_opt"] - links["p_nom"]

        init_cap_length = self.get_links_init_cap_length()
        new_cap_length = self.get_links_new_cap_length()

        links_capacities = pd.concat([links["p_nom"].rename('init [GW]'),
                                      links["p_nom_new"].rename('new [GW]'),
                                      init_cap_length.rename(columns={'init_cap_length': 'init [TWkm]'}),
                                      new_cap_length.rename(columns={'new_cap_length': 'new [TWkm]'})], axis=1)
        return links_capacities.round(2)

    # def get_lines_power(self):
    #     """countries_url_area_types the total power (MW) (in either direction) that goes through each type of
    #     line over self.net.snapshots"""
    #
    #     lines_t = self.net.lines_t
    #     lines_t.p0[lines_t.p0 < 0] = 0
    #     lines_t.p1[lines_t.p1 < 0] = 0
    #     power = lines_t.p0 + lines_t.p1
    #
    #     lines = self.net.lines
    #     carriers = sorted(list(set(lines.carrier.values)))
    #     power_carrier = dict.fromkeys(carriers)
    #     for carrier in carriers:
    #         lines_carrier = lines[lines.carrier == carrier]
    #         power_carrier[carrier] = power[lines_carrier.index].to_numpy().sum()
    #
        # return pd.DataFrame.from_dict(power_carrier, orient="index", columns=["lines_power"]).lines_power

    def get_links_power(self):
        """countries_url_area_types the total power (MW) (in either direction) that goes through all links over self.net.snapshots"""

        links_t = self.net.links_t
        links_carriers = self.net.links['carrier']

        carriers = links_carriers.unique()

        df_power = pd.Series(index=carriers)

        for carrier in carriers:

            links_to_keep = links_carriers[links_carriers == carrier]
            links_to_keep_t_p0 = links_t['p0'].loc[:, list(links_to_keep.index)]
            links_to_keep_t_p1 = links_t['p1'].loc[:, list(links_to_keep.index)]

            links_to_keep_t_p0[links_to_keep_t_p0 < 0] = 0
            links_to_keep_t_p1[links_to_keep_t_p1 < 0] = 0
            power = links_to_keep_t_p0 + links_to_keep_t_p0

            power_total = power.to_numpy().sum()
            df_power.loc[carrier] = power_total*(1e-3)

        return df_power.round(2)

    # def get_lines_usage(self):
    #     """countries_url_area_types the average transmission capacity usage of each type of line"""
    #
    #     _, _, opt_cap = self.get_lines_capacity()
    #     tot_power = self.get_lines_power()
    #     return tot_power/(opt_cap*len(self.net.snapshots))

    def get_links_usage(self):
        """countries_url_area_types the average transmission capacity usage of all links"""

        links_capacities = self.get_links_capacity()
        opt_capacities_GW = links_capacities['init [GW]'] + links_capacities['new [GW]']
        links_power = self.get_links_power()
        df_cf = (links_power*1e3)/(opt_capacities_GW*len(self.net.snapshots))

        return df_cf.round(3)

    # def get_lines_capex(self):
    #     """countries_url_area_types the capital expenses for building the new capacity for each type of line."""
    #
    #     lines = self.net.lines
    #     lines["s_nom_new"] = lines.s_nom_opt - lines.s_nom
    #     lines["capex"] = lines.s_nom_new*lines.capital_cost
    #
    #     return lines.groupby(["carrier"]).capex.sum()

    def get_links_capex(self):
        """countries_url_area_types the capital expenses for building the new capacity for all links."""

        links = self.net.links
        links["p_nom_new"] = links.p_nom_opt - links.p_nom
        links["capex"] = links.p_nom_new*links.capital_cost

        df_capex = links.groupby(["carrier"]).capex.sum()

        return df_capex.round(2)

    # def get_lines_length(self):
    #     return self.net.lines[["carrier", "length"]].groupby(["carrier"]).sum().length

    def get_links_length(self):
        return self.net.links[["carrier", "length"]].groupby(["carrier"]).sum().length

    def get_links_init_cap_length(self):
        self.net.links["init_cap_length"] = self.net.links.length*self.net.links.p_nom
        init_cap_length = self.net.links[["init_cap_length", "carrier"]].groupby("carrier").sum()
        return init_cap_length*(1e-3)

    def get_links_new_cap_length(self):
        self.net.links["new_cap_length"] = self.net.links.length*(self.net.links.p_nom_opt-self.net.links.p_nom)
        new_cap_length = self.net.links[["new_cap_length", "carrier"]].groupby("carrier").sum()
        return new_cap_length*(1e-3)

    # --- Storage --- #

    def display_storage(self):
        """Display information about storage"""

        print('\n\n\n# --- STORAGE --- #')

        cap_cost, marg_cost = self.get_storage_capital_and_marginal_cost()
        costs = pd.concat([cap_cost.rename('ccost'), marg_cost.rename('mcost'), self.get_storage_capex(), self.get_storage_opex()], axis=1)
        costs = costs.round(4)
        print(f"Costs (M$):\n{costs}\n")

        capacities_p = self.get_storage_power_capacity()
        capacities_e = self.get_storage_energy_capacity()
        capacities = pd.concat([capacities_p, capacities_e], axis=1)
        print(f"Storage capacities:\n{capacities}\n")

        storage_f = self.get_storage_energy_in()
        storage_e = capacities_e['init [GWh]'] + capacities_e['new [GWh]']
        cycles = (storage_f / storage_e).round(0)
        print(f"Storage cycles:\n{cycles}\n")

        spillage = self.get_storage_spillage()
        print(f"Storage spillage [GWh]:\n{spillage.round(2)}\n")


        # print(f"Storage power:\n{self.get_storage_power()}\n")
        # print(f"Storage energy:\n{self.get_storage_energy()}\n")
        # print(f"Storage power use:\n{self.get_storage_power_usage()}\n")
        # print(f"Storage energy use:\n{self.get_storage_energy_usage()}\n")

    def get_storage_capital_and_marginal_cost(self):

        su = self.net.storage_units.groupby(["type"])
        return su.capital_cost.mean(), su.marginal_cost.mean()

    def get_storage_power_capacity(self):
        """countries_url_area_types the original, new and optimal power capacities (in MW) for each type of storage unit."""

        storage_units = self.net.storage_units.groupby(["type"])
        init_capacities = storage_units.p_nom.sum()
        opt_capacities = storage_units.p_nom_opt.sum()
        new_capacities = opt_capacities - init_capacities

        capacities_p = pd.concat([init_capacities.rename('init [GW]'),
                                new_capacities.rename('new [GW]')], axis=1)

        return capacities_p.round(2)

    def get_storage_energy_capacity(self):
        """countries_url_area_types the original, new and optimal energy capacities (in MWh) for each type of storage unit."""

        storage_units = self.net.storage_units
        storage_units["p_nom_energy"] = storage_units.p_nom*storage_units.max_hours
        storage_units["p_nom_opt_energy"] = storage_units.p_nom_opt*storage_units.max_hours

        storage_units = storage_units.groupby(["type"])
        init_capacities = storage_units.p_nom_energy.sum()
        opt_capacities = storage_units.p_nom_opt_energy.sum()
        new_capacities = opt_capacities - init_capacities

        capacities_e = pd.concat([init_capacities.rename('init [GWh]'),
                                new_capacities.rename('new [GWh]')], axis=1)

        return capacities_e.round(2)

    def get_storage_power(self):
        """countries_url_area_types the total power (MW) that goes out or in of the battery."""

        storage_units = self.net.storage_units
        types = sorted(list(set(storage_units.type.values)))
        storage_units_t = self.net.storage_units_t

        power = dict.fromkeys(types)

        for tech_type in types:
            storage_units_type = storage_units[storage_units.type == tech_type]
            power_out = storage_units_t.p[storage_units_type.index].values
            power_out[power_out < 0] = 0
            power_out = power_out.sum()
            power_in = -storage_units_t.p[storage_units_type.index].values
            power_in[power_in < 0] = 0
            power_in = power_in.sum()
            power[tech_type] = power_out + power_in

        return pd.DataFrame.from_dict(power, orient="index", columns=["power"]).power

    def get_storage_energy_in(self):
        """countries_url_area_types the total energy (MWh) that is stored over self.net.snapshots."""

        storage_units = self.net.storage_units
        types = sorted(list(set(storage_units.type.values)))
        storage_units_t = self.net.storage_units_t
        storage_units_t.p[storage_units_t.p < 0.] = 0.

        energy = dict.fromkeys(types)

        for tech_type in types:
            storage_units_type = storage_units[storage_units.type == tech_type]
            energy[tech_type] = storage_units_t.p[storage_units_type.index].to_numpy().sum()

        return pd.DataFrame.from_dict(energy, orient="index", columns=["energy"]).energy



    def get_storage_spillage(self):

        storage_units = self.net.storage_units
        types = sorted(list(set(storage_units.type.values)))
        storage_units_t = self.net.storage_units_t

        energy = dict.fromkeys(types)

        for tech_type in types:
            storage_units_type = storage_units[storage_units.type == tech_type]
            energy[tech_type] = storage_units_t.spill[storage_units_type.index].to_numpy().sum()

        return pd.DataFrame.from_dict(energy, orient="index", columns=["energy"]).energy

    # def get_storage_power_usage(self):
    #     """Return the average power capacity usage of each type of storage unit."""
    #
    #     _, _, opt_cap = self.get_storage_power_capacity()
    #     tot_power = self.get_storage_power()
    #     return tot_power/(opt_cap*len(self.net.snapshots))

    # def get_storage_energy_usage(self):
    #     """Returns the average energy capacity usage of each type of storage unit"""
    #
    #     _, _, opt_cap = self.get_storage_energy_capacity()
    #     tot_power = self.get_storage_energy()
    #     return tot_power/(opt_cap*len(self.net.snapshots))

    def get_storage_opex(self):
        """Returns the capital expenses for building the new capacity for each type of storage unit."""

        storage_units = self.net.storage_units
        total_power = self.net.storage_units_t.p.abs().sum(axis=0)
        storage_units["opex"] = total_power * storage_units.marginal_cost

        return storage_units.groupby(["type"]).opex.sum()

    def get_storage_capex(self):
        """Returns the capital expenses for building the new capacity for each type of storage unit."""

        storage_units = self.net.storage_units
        storage_units["p_nom_new"] = storage_units.p_nom_opt - storage_units.p_nom
        storage_units["capex"] = storage_units.p_nom_new * storage_units.capital_cost

        return storage_units.groupby(["type"]).capex.sum()


    def display_co2(self):

        print('\n\n\n# --- CO2 --- #')

        df_co2 = pd.DataFrame(index=['CO2'], columns=['budget [Mt]', 'use [Mt]', 'use [%]'])

        generators_t = self.net.generators_t['p']
        generators_t_year = generators_t.sum(axis=0)

        generators_specs = self.net.generators[['carrier', 'efficiency']].copy()
        generators_emissions = self.net.carriers
        generators_specs['emissions'] = generators_specs['carrier'].map(generators_emissions['co2_emissions']).fillna(0.)
        generators_specs['emissions_eff'] = generators_specs['emissions']/generators_specs['efficiency']

        co2_emissions_per_gen = generators_t_year * generators_specs['emissions_eff'] * 1e-3
        co2_emissions = co2_emissions_per_gen.sum()

        co2_budget = self.net.global_constraints.constant.values[0] * (1e-3)

        df_co2.loc['CO2', 'budget [Mt]'] = co2_budget
        df_co2.loc['CO2', 'use [Mt]'] = co2_emissions
        df_co2.loc['CO2', 'use [%]'] = co2_emissions / co2_budget

        df_co2 = df_co2.round(2)

        print(f"CO2 utilization:\n{df_co2}\n")





if __name__ == "__main__":

    topology = 'tyndp2018'

    run_id = '20200429_150213'

    main_output_dir = f'../../output/sizing/{topology}/'
    output_dir = f"{main_output_dir}{run_id}/"

    net = Network()
    net.import_from_csv_folder(output_dir)

    pprp = SizingResults(net)

    pprp.display_generation()
    pprp.display_transmission()
    pprp.display_storage()
    pprp.display_co2()