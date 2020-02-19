from pypsa import Network
import pandas as pd


class PyPSAResults:

    """
    This class allows to extract results from a optimized PyPSA network.
    """

    def __init__(self, network: Network):

        self.net = network

    # --- Generation --- #

    def display_generation(self):
        """Display information about generation"""

        print(self.get_generators_capacity())
        print(self.get_generators_generation())
        print(self.get_generators_average_usage())
        print(self.get_generators_opex())
        print(self.get_generators_capex())
        print(self.get_generators_cost())

    def get_generators_capacity(self):
        """Returns the original, new and optimal generation capacities (in MW) for each type of generator."""

        gens = self.net.generators.groupby(["carrier"])
        init_capacities = gens.p_nom.sum()
        opt_capacities = gens.p_nom_opt.sum()
        new_capacities = opt_capacities - init_capacities

        return init_capacities, new_capacities, opt_capacities

    def get_generators_generation(self):
        """Returns the total generation (in MWh) over the self.net.snapshots for each type of generator."""

        gens = self.net.generators
        carriers = sorted(list(set(gens.carrier.values)))
        gens_t = self.net.generators_t

        generation = dict.fromkeys(carriers)

        for carrier in carriers:
            gens_carrier = gens[gens.carrier == carrier]
            generation[carrier] = gens_t.p[gens_carrier.index].to_numpy().sum()

        return pd.DataFrame.from_dict(generation, orient="index", columns=["generation"]).generation

    def get_generators_average_usage(self):
        """Returns the average generation capacity usage (i.e. mean(generation_t/capacity)) of each type of generator"""

        _, _, opt_cap = self.get_generators_capacity()
        tot_gen = self.get_generators_generation()
        return tot_gen/(opt_cap*len(self.net.snapshots))

    def get_generators_opex(self):
        """Returns the operational expenses of running each type of generator over the self.net.snapshots"""

        gens = self.net.generators
        carriers = sorted(list(set(gens.carrier.values)))
        gens_t = self.net.generators_t

        opex = dict.fromkeys(carriers)

        for carrier in carriers:
            gens_carrier = gens[gens.carrier == carrier]
            generation_per_gen = gens_t.p[gens_carrier.index]*gens_carrier.marginal_cost
            opex[carrier] = generation_per_gen.to_numpy().sum()

        return pd.DataFrame.from_dict(opex, orient="index", columns=["opex"]).opex

    # TODO: need to check that the units are correct
    def get_generators_capex(self):
        """Returns the capital expenses for building the new capacity for each type of generator."""

        gens = self.net.generators
        gens["p_nom_new"] = gens.p_nom_opt - gens.p_nom
        gens["capex"] = gens.p_nom_new*gens.capital_cost

        return gens.groupby(["carrier"]).capex.sum()

    def get_generators_cost(self):
        return self.get_generators_opex() + self.get_generators_capex()

    # --- Transmission --- #

    def display_transmission(self):
        """Display information about transmission"""

        print(self.get_lines_capacity())
        print(self.get_lines_power())
        print(self.get_lines_usage())
        print(self.get_lines_capex())

    def get_lines_capacity(self):
        """Returns the original, new and optimal transmission capacities (in MW) for each type of line."""

        lines = self.net.lines.groupby(["carrier"])
        init_capacities = lines.s_nom.sum()
        opt_capacities = lines.s_nom_opt.sum()
        new_capacities = opt_capacities - init_capacities

        return init_capacities, new_capacities, opt_capacities

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

    def get_lines_usage(self):
        """Returns the average transmission capacity usage of each type of line"""

        _, _, opt_cap = self.get_lines_capacity()
        tot_power = self.get_lines_power()
        return tot_power/(opt_cap*len(self.net.snapshots))

    def get_lines_capex(self):
        """Returns the capital expenses for building the new capacity for each type of line."""

        lines = self.net.lines
        lines["s_nom_new"] = lines.s_nom_opt - lines.s_nom
        lines["capex"] = lines.s_nom_new*lines.capital_cost

        return lines.groupby(["carrier"]).capex.sum()

    # --- Storage --- #

    def display_storage(self):
        """Display information about storage"""

        print(self.get_storage_power_capacity())
        print(self.get_storage_energy_capacity())
        print(self.get_storage_power())
        print(self.get_storage_energy())
        print(self.get_storage_power_usage())
        print(self.get_storage_energy_usage())
        print(self.get_lines_capex())

    def get_storage_power_capacity(self):
        """Returns the original, new and optimal power capacities (in MW) for each type of storage unit."""

        storage_units = self.net.storage_units.groupby(["carrier"])
        init_capacities = storage_units.p_nom.sum()
        opt_capacities = storage_units.p_nom_opt.sum()
        new_capacities = opt_capacities - init_capacities

        return init_capacities, new_capacities, opt_capacities

    def get_storage_energy_capacity(self):
        """Returns the original, new and optimal energy capacities (in MWh) for each type of storage unit."""

        storage_units = self.net.storage_units
        storage_units["p_nom_energy"] = storage_units.p_nom*storage_units.max_hours
        storage_units["p_nom_opt_energy"] = storage_units.p_nom_opt*storage_units.max_hours

        storage_units = storage_units.groupby(["carrier"])
        init_capacities = storage_units.p_nom_energy.sum()
        opt_capacities = storage_units.p_nom_opt_energy.sum()
        new_capacities = opt_capacities - init_capacities

        return init_capacities, new_capacities, opt_capacities

    def get_storage_power(self):
        """Returns the total power (MW) that goes out or in of the battery"""

        storage_units = self.net.storage_units
        carriers = sorted(list(set(storage_units.carrier.values)))
        storage_units_t = self.net.storage_units_t

        power = dict.fromkeys(carriers)

        for carrier in carriers:
            storage_units_carrier = storage_units[storage_units.carrier == carrier]
            power_out = storage_units_t.p[storage_units_carrier.index].values
            power_out[power_out < 0] = 0
            power_out = power_out.sum()
            power_in = -storage_units_t.p[storage_units_carrier.index].values
            power_in[power_in < 0] = 0
            power_in = power_in.sum()
            power[carrier] = power_out + power_in

        return pd.DataFrame.from_dict(power, orient="index", columns=["power"]).power

    def get_storage_energy(self):
        """Returns the total energy (MWh) that is stored over self.net.snapshots"""

        storage_units = self.net.storage_units
        carriers = sorted(list(set(storage_units.carrier.values)))
        storage_units_t = self.net.storage_units_t

        energy = dict.fromkeys(carriers)

        for carrier in carriers:
            storage_units_carrier = storage_units[storage_units.carrier == carrier]
            energy[carrier] = storage_units_t.state_of_charge[storage_units_carrier.index].to_numpy().sum()

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

    def get_storage_capex(self):
        """Returns the capital expenses for building the new capacity for each type of storage unit."""

        storage_units = self.net.storage_units
        storage_units["p_nom_new"] = storage_units.p_nom_opt - storage_units.p_nom
        storage_units["capex"] = storage_units.p_nom_new * storage_units.capital_cost

        return storage_units.groupby(["carrier"]).capex.sum()
