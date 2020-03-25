import sys
import os

import pandas as pd

from pypsa import Network

from src.parameters.costs import get_cost


class SizingResults:

    """
    This class allows to extract results from a optimized PyPSA network.
    """

    def __init__(self, network: Network):

        self.net = network

    # --- Generation --- #

    def display_generation(self):
        """Display information about generation"""

        cap_cost, marg_cost = self.get_gen_capital_and_marginal_cost()
        print(f"Capital cost:\n{cap_cost}\n")
        print(f"Marginal cost:\n{marg_cost}\n")
        init_capacities, new_capacities, opt_capacities, max_capacities = self.get_generators_capacity()
        print(f"Generators capacity:\nInit:\n{init_capacities}\nNew:\n{new_capacities}\nTotal:\n{opt_capacities}\n"
              f"Max:\n{max_capacities}\n")
        print(f"Generators generation:\n{self.get_generators_generation()}\n")
        print(f"Total generation:\n{self.get_generators_generation().sum()}\n")
        print(f"Number of generators:\n{self.get_generators_numbers()}\n")
        print(f"Generators Average Use:\n{self.get_generators_average_usage()}\n")
        print(f"Generators Opex:\n{self.get_generators_opex()}\n")
        print(f"Generators Capex:\n{self.get_generators_capex()}\n")
        print(f"Generators Cost:\n{self.get_generators_cost()}\n")

    def get_gen_capital_and_marginal_cost(self):

        gens = self.net.generators.groupby(["type"])
        return gens.capital_cost.mean(), gens.marginal_cost.mean()

    def get_generators_capacity(self):
        """Returns the original, new and optimal generation capacities (in MW) for each type of generator."""

        gens = self.net.generators.groupby(["type"])
        init_capacities = gens.p_nom.sum()
        opt_capacities = gens.p_nom_opt.sum()
        max_capacities = gens.p_nom_max.sum()
        new_capacities = opt_capacities - init_capacities

        return init_capacities, new_capacities, opt_capacities, max_capacities

    def get_generators_numbers(self):

        return self.net.generators.groupby("type").count().bus

    def get_generators_generation(self):
        """Returns the total generation (in MWh) over the self.net.snapshots for each type of generator."""

        gens = self.net.generators
        types = sorted(list(set(gens.type.values)))
        gens_t = self.net.generators_t

        generation = dict.fromkeys(types)

        for tech_type in types:
            gens_type = gens[gens.type == tech_type]
            generation[tech_type] = gens_t.p[gens_type.index].to_numpy().sum()

        return pd.DataFrame.from_dict(generation, orient="index", columns=["generation"]).generation

    def get_generators_average_usage(self):
        """Returns the average generation capacity usage (i.e. mean(generation_t/capacity)) of each type of generator"""

        _, _, opt_cap, _ = self.get_generators_capacity()
        tot_gen = self.get_generators_generation()
        return tot_gen/(opt_cap*len(self.net.snapshots))

    def get_generators_opex(self):
        """Returns the operational expenses of running each type of generator over the self.net.snapshots"""

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
        """Returns the capital expenses for building the new capacity for each type of generator."""

        gens = self.net.generators
        gens["p_nom_new"] = gens.p_nom_opt - gens.p_nom
        gens["capex"] = gens.p_nom_new*gens.capital_cost

        return gens.groupby(["type"]).capex.sum()

    def get_generators_cost(self):
        return self.get_generators_opex() + self.get_generators_capex()

    # --- Transmission --- #

    def display_transmission(self):
        """Display information about transmission"""

        print(f"AC cost (M€/GW/km):  {get_cost('AC', len(self.net.snapshots))}")
        print(f"DC cost (M€/GW/km): {get_cost('DC', len(self.net.snapshots))}\n")

        if len(self.net.lines.index) != 0:
            init_capacities, new_capacities, opt_capacities = self.get_lines_capacity()
            print(f"Lines capacity:\nInit:\n{init_capacities}\nNew:\n{new_capacities}\nTotal:\n{opt_capacities}\n")
            print(f"Lines power:\n{self.get_lines_power()}\n")
            print(f"Lines use:\n{self.get_lines_usage()}\n")
            print(f"Lines capex:\n{self.get_lines_capex()}\n")
        if len(self.net.links.index) != 0:
            init_capacities, new_capacities, opt_capacities = self.get_links_capacity()
            print(f"Links capacity:\nInit:\n{init_capacities}\nNew:\n{new_capacities}\nTotal:\n{opt_capacities}\n")
            print(f"Links power:\n{self.get_links_power()}\n")
            print(f"Links use:\n{self.get_links_usage()}\n")
            print(f"Links capex:\n{self.get_links_capex()}\n")
            # print(f"Links length:\n{self.get_links_length()}\n")
            print(f"Links init cap*length:\n{self.get_links_init_cap_length()}\n")
            print(f"Links new cap*length:\n{self.get_links_new_cap_length()}\n")

    def get_lines_capacity(self):
        """Returns the original, new and optimal transmission capacities (in MW) for each type of line."""

        lines = self.net.lines.groupby(["carrier"])
        init_capacities = lines.s_nom.sum()
        opt_capacities = lines.s_nom_opt.sum()
        new_capacities = opt_capacities - init_capacities

        return init_capacities, new_capacities, opt_capacities

    def get_links_capacity(self):
        """Returns the original, new and optimal transmission capacities (in MW) for links."""

        links = self.net.links[["carrier", "p_nom", "p_nom_opt"]].groupby("carrier").sum()
        links["p_nom_new"] = links["p_nom_opt"] - links["p_nom"]
        return links["p_nom"], links["p_nom_new"], links["p_nom_opt"]

    def get_lines_power(self):
        """Returns the total power (MW) (in either direction) that goes through each type of
        line over self.net.snapshots"""

        lines_t = self.net.lines_t
        lines_t.p0[lines_t.p0 < 0] = 0
        lines_t.p1[lines_t.p1 < 0] = 0
        power = lines_t.p0 + lines_t.p1

        lines = self.net.lines
        carriers = sorted(list(set(lines.carrier.values)))
        power_carrier = dict.fromkeys(carriers)
        for carrier in carriers:
            lines_carrier = lines[lines.carrier == carrier]
            power_carrier[carrier] = power[lines_carrier.index].to_numpy().sum()

        return pd.DataFrame.from_dict(power_carrier, orient="index", columns=["lines_power"]).lines_power

    def get_links_power(self):
        """Returns the total power (MW) (in either direction) that goes through all links over self.net.snapshots"""

        links_t = self.net.links_t
        links_t.p0[links_t.p0 < 0] = 0
        links_t.p1[links_t.p1 < 0] = 0
        power = links_t.p0 + links_t.p1

        return power.to_numpy().sum()

    def get_lines_usage(self):
        """Returns the average transmission capacity usage of each type of line"""

        _, _, opt_cap = self.get_lines_capacity()
        tot_power = self.get_lines_power()
        return tot_power/(opt_cap*len(self.net.snapshots))

    def get_links_usage(self):
        """Returns the average transmission capacity usage of all links"""

        _, _, opt_cap = self.get_links_capacity()
        tot_power = self.get_links_power()
        return tot_power/(opt_cap*len(self.net.snapshots))

    def get_lines_capex(self):
        """Returns the capital expenses for building the new capacity for each type of line."""

        lines = self.net.lines
        lines["s_nom_new"] = lines.s_nom_opt - lines.s_nom
        lines["capex"] = lines.s_nom_new*lines.capital_cost

        return lines.groupby(["carrier"]).capex.sum()

    def get_links_capex(self):
        """Returns the capital expenses for building the new capacity for all links."""

        links = self.net.links
        links["p_nom_new"] = links.p_nom_opt - links.p_nom
        links["capex"] = links.p_nom_new*links.capital_cost

        return links.groupby(["carrier"]).capex.sum()

    def get_lines_length(self):
        return self.net.lines[["carrier", "length"]].groupby(["carrier"]).sum().length

    def get_links_length(self):
        return self.net.links[["carrier", "length"]].groupby(["carrier"]).sum().length

    def get_links_init_cap_length(self):
        print(len(self.net.links))
        print(len(self.net.buses))
        self.net.links["init_cap_length"] = self.net.links.length*self.net.links.p_nom
        return self.net.links[["init_cap_length", "carrier"]].groupby("carrier").sum()

    def get_links_new_cap_length(self):
        self.net.links["new_cap_length"] = self.net.links.length*(self.net.links.p_nom_opt-self.net.links.p_nom)
        return self.net.links[["new_cap_length", "carrier"]].groupby("carrier").sum()

    # --- Storage --- #

    def display_storage(self):
        """Display information about storage"""

        cap_cost, marg_cost = self.get_storage_capital_and_marginal_cost()
        print(f"Capital cost:\n{cap_cost}\n")
        print(f"Marginal cost:\n{marg_cost}\n")
        init_capacities, new_capacities, opt_capacities = self.get_storage_power_capacity()
        print(f"Storage power capacity:\nInit:\n{init_capacities}\nNew:\n{new_capacities}\nTotal:\n{opt_capacities}\n")
        init_capacities, new_capacities, opt_capacities = self.get_storage_energy_capacity()
        print(f"Storage energy capacity:\nInit:\n{init_capacities}\nNew:\n{new_capacities}\nTotal:\n{opt_capacities}\n")
        print(f"Storage power:\n{self.get_storage_power()}\n")
        print(f"Storage energy:\n{self.get_storage_energy()}\n")
        print(f"Storage power use:\n{self.get_storage_power_usage()}\n")
        print(f"Storage energy use:\n{self.get_storage_energy_usage()}\n")
        print(f"Storage opex:\n{self.get_storage_opex()}\n")
        print(f"Storage capex:\n{self.get_storage_capex()}\n")

    def get_storage_capital_and_marginal_cost(self):

        su = self.net.storage_units.groupby(["type"])
        return su.capital_cost.mean(), su.marginal_cost.mean()

    def get_storage_power_capacity(self):
        """Returns the original, new and optimal power capacities (in MW) for each type of storage unit."""

        storage_units = self.net.storage_units.groupby(["type"])
        init_capacities = storage_units.p_nom.sum()
        opt_capacities = storage_units.p_nom_opt.sum()
        new_capacities = opt_capacities - init_capacities

        return init_capacities, new_capacities, opt_capacities

    def get_storage_energy_capacity(self):
        """Returns the original, new and optimal energy capacities (in MWh) for each type of storage unit."""

        storage_units = self.net.storage_units
        storage_units["p_nom_energy"] = storage_units.p_nom*storage_units.max_hours
        storage_units["p_nom_opt_energy"] = storage_units.p_nom_opt*storage_units.max_hours

        storage_units = storage_units.groupby(["type"])
        init_capacities = storage_units.p_nom_energy.sum()
        opt_capacities = storage_units.p_nom_opt_energy.sum()
        new_capacities = opt_capacities - init_capacities

        return init_capacities, new_capacities, opt_capacities

    def get_storage_power(self):
        """Returns the total power (MW) that goes out or in of the battery"""

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

    def get_storage_energy(self):
        """Returns the total energy (MWh) that is stored over self.net.snapshots"""

        storage_units = self.net.storage_units
        types = sorted(list(set(storage_units.type.values)))
        storage_units_t = self.net.storage_units_t

        energy = dict.fromkeys(types)

        for tech_type in types:
            storage_units_type = storage_units[storage_units.type == tech_type]
            energy[tech_type] = storage_units_t.state_of_charge[storage_units_type.index].to_numpy().sum()

        return pd.DataFrame.from_dict(energy, orient="index", columns=["energy"]).energy

    def get_storage_power_usage(self):
        """Returns the average power capacity usage of each type of storage unit"""

        _, _, opt_cap = self.get_storage_power_capacity()
        tot_power = self.get_storage_power()
        return tot_power/(opt_cap*len(self.net.snapshots))

    def get_storage_energy_usage(self):
        """Returns the average energy capacity usage of each type of storage unit"""

        _, _, opt_cap = self.get_storage_energy_capacity()
        tot_power = self.get_storage_energy()
        return tot_power/(opt_cap*len(self.net.snapshots))

    def get_storage_opex(self):
        """Returns the capital expenses for building the new capacity for each type of storage unit."""

        storage_units = self.net.storage_units
        total_power = self.net.storage_units_t.p.abs().sum(axis=0)
        storage_units["capex"] = total_power * storage_units.marginal_cost

        return storage_units.groupby(["type"]).capex.sum()

    def get_storage_capex(self):
        """Returns the capital expenses for building the new capacity for each type of storage unit."""

        storage_units = self.net.storage_units
        storage_units["p_nom_new"] = storage_units.p_nom_opt - storage_units.p_nom
        storage_units["capex"] = storage_units.p_nom_new * storage_units.capital_cost

        return storage_units.groupby(["type"]).capex.sum()


if __name__ == "__main__":

    assert (len(sys.argv) == 2) or (len(sys.argv) == 3), \
        "You need to provide one or two argument: output_dir (and test_number)"

    main_output_dir = sys.argv[1]
    test_number = sys.argv[2] if len(sys.argv) == 3 else None
    if test_number is None:
        test_number = sorted(os.listdir(main_output_dir))[-1]
    output_dir = main_output_dir + test_number + "/"
    print(output_dir)
    net = Network()
    net.import_from_csv_folder(output_dir)

    print(f"CO2 limit (kT):\n{net.global_constraints.constant}\n")
    print(f"Total load (GWh):\n{net.loads_t.p.values.sum()}\n")
    print(f"Total ccgt capacity\n{net.generators[net.generators.type == 'ccgt'].p_nom_opt.sum()}")
    print(f"Number of res sites\n"
          f"{len(net.generators[net.generators.type.isin(['wind_onshore', 'wind_offshore', 'pv_utility', 'pv_residential'])])}")

    pprp = SizingResults(net)
    pprp.display_generation()
    # pprp.display_transmission()
    # pprp.display_storage()
