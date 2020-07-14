import sys
from os import listdir
import pickle

import pandas as pd

from src.resite.resite import Resite


class ResiteResults:

    def __init__(self, resite: Resite):
        self.resite = resite
        self.data = self.resite.data_dict
        self.sel_data = self.resite.sel_data_dict
        self.existing_nodes = self.data["existing_cap_ds"][self.data["existing_cap_ds"] > 0].index
        self.optimal_cap_ds = self.resite.y_ds*self.data["cap_potential_ds"]

    def print_summary(self):
        print(f"\nRegion: {self.resite.regions}")
        print(f"Technologies: {self.resite.technologies}")
        print(f"Formulation: {self.resite.formulation}")
        print(f"Formulation parameters: {self.resite.formulation_params}\n")

    def get_initial_points_number(self):
        count = pd.Series(0, index=sorted(list(self.resite.tech_points_dict.keys())), dtype=int)
        for tech, points in self.resite.tech_points_dict.items():
            count[tech] = int(len(points))
        return count

    def get_selected_points_number(self):
        count = pd.Series(0, index=sorted(list(self.resite.sel_tech_points_dict.keys())), dtype=int)
        for tech, points in self.resite.sel_tech_points_dict.items():
            count[tech] = int(len(points))
        return count

    def get_existing_points_number(self):
        count = pd.Series(0, index=sorted(list(set(self.existing_nodes.droplevel(1)))), dtype=int)
        for tech, point in self.existing_nodes:
            count.loc[tech] += 1
        return count

    def get_new_points_number(self):
        return self.get_selected_points_number() - self.get_existing_points_number()

    def print_number_of_points(self):
        count = pd.concat([self.get_initial_points_number(), self.get_selected_points_number(),
                           self.get_existing_points_number()], axis=1, sort=True)
        count.columns = ["Initial", "Selected", "With existing cap"]
        print(f"Number of points:\n{count}\n")

    def get_initial_capacity_potential_sum(self):
        return self.data["cap_potential_ds"].groupby(level=0).sum()

    def get_selected_capacity_potential_sum(self):
        return self.sel_data["cap_potential_ds"].groupby(level=0).sum()

    def get_initial_capacity_potential_mean(self):
        return self.data["cap_potential_ds"].groupby(level=0).mean()

    def get_selected_capacity_potential_mean(self):
        return self.sel_data["cap_potential_ds"].groupby(level=0).mean()

    def get_initial_capacity_potential_std(self):
        return self.data["cap_potential_ds"].groupby(level=0).std()

    def get_selected_capacity_potential_std(self):
        return self.sel_data["cap_potential_ds"].groupby(level=0).std()

    def print_capacity_potential(self):
        initial_cap_potential = self.get_initial_capacity_potential_sum()
        selected_cap_potential = self.get_selected_capacity_potential_sum()
        cap_potential = pd.concat([initial_cap_potential, selected_cap_potential], axis=1, sort=True)
        cap_potential.columns = ["Initial", "Selected"]
        print(f"Capacity potential (GW):\n{cap_potential}\n")

    def get_selected_capacity_potential_use(self):
        potential_capacity_use = self.optimal_cap_ds/self.sel_data["cap_potential_ds"]
        potential_capacity_use = potential_capacity_use.dropna()
        return potential_capacity_use.groupby(level=0).mean()

    def get_existing_capacity(self):
        return self.data["existing_cap_ds"].groupby(level=0).sum()

    def get_optimal_capacity(self):
        return self.optimal_cap_ds.groupby(level=0).sum()

    def get_new_capacity(self):
        return self.optimal_cap_ds.groupby(level=0).sum() - \
               self.data["existing_cap_ds"].groupby(level=0).sum()

    def get_optimal_capacity_at_existing_nodes(self):
        return self.optimal_cap_ds[self.existing_nodes].groupby(level=0).sum()

    def print_capacity(self):
        existing_cap = self.get_existing_capacity()
        optimal_cap = self.get_optimal_capacity()
        optimal_cap_at_ex_nodes = self.get_optimal_capacity_at_existing_nodes()
        capacities = pd.concat([existing_cap, optimal_cap, optimal_cap_at_ex_nodes], axis=1, sort=True)
        capacities.columns = ["Existing", "Optimal", "Optimal at existing nodes"]
        print(f"Capacity (GW):\n{capacities}\n")

    def get_generation(self):
        generation = self.optimal_cap_ds * self.data["cap_factor_df"]
        return generation.sum().groupby(level=0).sum()

    def print_generation(self):
        generation = self.optimal_cap_ds*self.data["cap_factor_df"]
        generation_per_type = pd.DataFrame(generation.sum().groupby(level=0).sum(), columns=["GWh"])
        generation_per_type["% of Total"] = generation_per_type["GWh"]/generation_per_type["GWh"].sum()
        generation_per_type["At Existing Nodes"] = generation[self.existing_nodes].sum().groupby(level=0).sum()
        print(f"Generation (GWh):\n{generation_per_type}\n")

    def get_initial_cap_factor_mean(self):
        return self.data["cap_factor_df"].mean().groupby(level=0).mean()

    def get_selected_cap_factor_mean(self):
        return self.sel_data["cap_factor_df"].mean().groupby(level=0).mean()

    def get_unselected_cap_factor_mean(self):
        cap_factor_df = self.data["cap_factor_df"].drop(self.sel_data["cap_factor_df"].columns, axis=1)
        return cap_factor_df.mean().groupby(level=0).mean()

    def print_cap_factor_mean(self):
        initial_cap_factor_mean = self.get_initial_cap_factor_mean()
        selected_cap_factor_mean = self.get_selected_cap_factor_mean()
        cap_factor_mean = pd.concat([initial_cap_factor_mean, selected_cap_factor_mean], axis=1, sort=True)
        cap_factor_mean.columns = ["Initial", "Selected"]
        print(f"Mean of mean of capacity factors:\n{cap_factor_mean}\n")

    def get_initial_cap_factor_median(self):
        return self.data["cap_factor_df"].median().groupby(level=0).mean()

    def get_selected_cap_factor_median(self):
        return self.sel_data["cap_factor_df"].median().groupby(level=0).mean()

    def get_unselected_cap_factor_median(self):
        cap_factor_df = self.data["cap_factor_df"].drop(self.sel_data["cap_factor_df"].columns, axis=1)
        return cap_factor_df.median().groupby(level=0).mean()

    def print_cap_factor_median(self):
        initial_cap_factor_median = self.get_initial_cap_factor_median()
        selected_cap_factor_median = self.get_selected_cap_factor_median()
        cap_factor_median = pd.concat([initial_cap_factor_median, selected_cap_factor_median], axis=1, sort=True)
        cap_factor_median.columns = ["Initial", "Selected"]
        print(f"Mean of mean of capacity factors:\n{cap_factor_median}\n")

    def get_initial_cap_factor_std(self):
        return self.data["cap_factor_df"].std().groupby(level=0).mean()

    def get_selected_cap_factor_std(self):
        return self.sel_data["cap_factor_df"].std().groupby(level=0).mean()

    def get_unselected_cap_factor_std(self):
        cap_factor_df = self.data["cap_factor_df"].drop(self.sel_data["cap_factor_df"].columns, axis=1)
        return cap_factor_df.std().groupby(level=0).mean()

    def print_cap_factor_std(self):
        initial_cap_factor_std = self.get_initial_cap_factor_std()
        selected_cap_factor_std = self.get_selected_cap_factor_std()
        cap_factor_std = pd.concat([initial_cap_factor_std, selected_cap_factor_std], axis=1, sort=True)
        cap_factor_std.columns = ["Initial", "Selected"]
        print(f"Mean of std of capacity factors:\n{cap_factor_std}\n")


if __name__ == "__main__":

    assert (len(sys.argv) == 2) or (len(sys.argv) == 3), \
        "You need to provide one or two argument: output_dir (and test_number)"

    main_output_dir = sys.argv[1]
    test_number = sys.argv[2] if len(sys.argv) == 3 else None
    if test_number is None:
        test_number = sorted(listdir(main_output_dir))[-1]
    output_dir = f"{main_output_dir}{test_number}/"
    print(output_dir)

    resite_ = pickle.load(open(f"{output_dir}resite_instance.p", 'rb'))

    ro = ResiteResults(resite_)
    ro.print_summary()

    ro.print_number_of_points()

    ro.print_capacity()
    ro.print_capacity_potential()

    ro.print_cap_factor_mean()
    ro.print_cap_factor_std()

    ro.print_generation()
