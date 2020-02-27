import pickle
import numpy as np
import pandas as pd
import yaml


def get_nb_generators(generators, types):
    dict_gens = dict.fromkeys(types)
    for t in types:
        dict_gens[t] = len(generators.where(generators.type == t, drop=True).id)
    return dict_gens


def total_load(loads):
    return np.sum(loads.p_set.values)


def total_generation(generators):
    return np.sum(generators.p.values)


def get_lost_load(loads, generators):
    load = np.sum(loads.p_set.values)
    generation = np.sum(generators.p.values)
    charge = np.sum(n.storages.charge.values)
    return load + charge - generation


def get_curtailment(generators):
    generated = generators.p.values
    max_generation_per_time = generators.p_nom_opt*generators.p_max_pu
    return np.sum(max_generation_per_time.values - generated)


def get_curtailment_per_type(generators, types):
    curtail_dict = dict.fromkeys(types)
    for t in types:
        generated = generators.where(generators.type == t, drop=True).p.values
        max_generation_per_time = generators.where(generators.type == t, drop=True).p_nom_opt * \
            generators.where(generators.type == t, drop=True).p_max_pu
        curtail_dict[t] = np.sum(max_generation_per_time - generated).item()
    return curtail_dict


def get_storage_energy_cap(storages):
    return np.sum(storages.p_nom_opt*storages.max_hours).item()


def get_storage_energy(storages):
    return np.sum(storages.state_of_charge).item()


def get_total_trans_cap(lines):
    return np.sum(lines.s_nom_opt).item()


def get_new_trans_cap(lines):
    return np.sum(lines.s_nom_opt-lines.s_nom).item()


def get_max_trans(lines):
    return np.max(lines.s_nom_opt.values)


def get_length(lines):
    return np.sum(lines.length)


def get_generation(generators, types):
    generation_dict = dict.fromkeys(types)
    for t in types:
        generation_dict[t] = np.sum(generators.where(generators.type == t, drop=True).p.values)
    return generation_dict


def get_max_capacity(generators, types):
    capacity_dict = dict.fromkeys(types)
    for t in types:
        capacity_dict[t] = np.sum(generators.where(generators.type == t, drop=True).p_nom_max.values)
    return capacity_dict


def get_opt_capacity(generators, types):
    capacity_dict = dict.fromkeys(types)
    for t in types:
        capacity_dict[t] = np.sum(generators.where(generators.type == t, drop=True).p_nom_opt.values)
    return capacity_dict


def get_gen_capex_costs(generators, types):
    cost_dict = dict.fromkeys(types)
    for t in types:
        cost_dict[t] = np.mean(generators.where(generators.type == t, drop=True).capital_cost.values)
    return cost_dict


def get_gen_opex_costs(generators, types):
    cost_dict = dict.fromkeys(types)
    for t in types:
        cost_dict[t] = np.mean(generators.where(generators.type == t, drop=True).marginal_cost.values)
    return cost_dict


def get_total_gen_capex_costs(generators, types):
    cost_dict = dict.fromkeys(types)
    for t in types:
        gens = generators.where(generators.type == t, drop=True)
        cost_dict[t] = np.sum(gens.capital_cost*(gens.p_nom_opt-gens.p_nom)).item()
    return cost_dict


def get_total_gen_opex_costs(generators, types):
    cost_dict = dict.fromkeys(types)
    for t in types:
        gens = generators.where(generators.type == t, drop=True)
        print(len(gens.capital_cost))
        print(len(gens.marginal_cost*np.sum(gens.p, axis=1)))
        cost_dict[t] = np.sum(gens.marginal_cost*np.sum(gens.p, axis=1)).item()
    return cost_dict


runs = ['20191214_135908', '20191214_140610', '20191214_140624', '20191214_140641',
        '20191214_140652', '20191214_140809', '20191214_230851']
names = ['no_siting', '200', '250', '300', '400', '400_co2', '400_max']
max_capacities = pd.DataFrame(index=names, columns=["wind", "pv"])
opt_capacities = pd.DataFrame(index=names, columns=["ccgt", "wind", "pv", "trans-tot", "trans-new", "store"])
total_gen = pd.DataFrame(index=names, columns=["wind", "pv"])
lost_load_and_curtailment = pd.DataFrame(index=names, columns=["lost load", "wind curt", "pv curt"])
tech_unit_capex_costs = pd.DataFrame(index=names, columns=["ccgt", "wind", "pv", "trans", "store"])
tech_unit_opex_costs = pd.DataFrame(index=names, columns=["ccgt", "wind", "pv", "store"])
tech_capex_costs = pd.DataFrame(index=names, columns=["ccgt", "wind", "pv", "trans", "store", "total"])
tech_opex_costs = pd.DataFrame(index=names, columns=["ccgt", "wind", "pv", "trans", "store", "total"])
lost_load_costs = pd.DataFrame(index=names, columns=["total"])
total_costs = pd.DataFrame(index=names, columns=["total"])
total_lines = pd.DataFrame(index=names, columns=["total"])
new_lines = pd.DataFrame(index=names, columns=["total"])

for i, run in enumerate(runs):
    print(names[i])
    # Read the network
    n = pickle.load(open("../output/examples/e-highways/" + run + "/optimized_network.pkl", 'rb'))
    costs = yaml.safe_load(open("../output/examples/e-highways/" + run + "/costs.yaml", 'r'))

    # print("Total number of generators", len(n.generators.id))
    # print("Number of generators", get_nb_generators(n.generators, ["ccgt", "solar_tallmaxm", "wind_aeodyn"]))
    #
    # print("Total load", total_load(n.loads))

    # Get max capacities
    gen_max = get_max_capacity(n.generators, ["solar_tallmaxm", "wind_aerodyn"])
    max_capacities.loc[names[i]]["solar"] = gen_max["solar_tallmaxm"]
    max_capacities.loc[names[i]]["wind"] = gen_max["wind_aerodyn"]

    # Get opt capacities
    gen_opt = get_opt_capacity(n.generators, ["ccgt", "solar_tallmaxm", "wind_aerodyn"])
    opt_capacities.loc[names[i]]["ccgt"] = gen_opt["ccgt"]
    opt_capacities.loc[names[i]]["solar"] = gen_opt["solar_tallmaxm"]
    opt_capacities.loc[names[i]]["wind"] = gen_opt["wind_aerodyn"]
    opt_capacities.loc[names[i]]["trans-tot"] = get_total_trans_cap(n.lines)
    opt_capacities.loc[names[i]]["trans-new"] = get_new_trans_cap(n.lines)
    opt_capacities.loc[names[i]]["store"] = get_storage_energy_cap(n.storages)

    # Get generation
    gen_gen = get_generation(n.generators, ["ccgt", "solar_tallmaxm", "wind_aerodyn"])
    print(gen_gen)
    total_gen.loc[names[i]]["ccgt"] = gen_gen["ccgt"]
    total_gen.loc[names[i]]["solar"] = gen_gen["solar_tallmaxm"]
    total_gen.loc[names[i]]["wind"] = gen_gen["wind_aerodyn"]

    # Get lost load and curtailment
    lost_load_and_curtailment.loc[names[i]]["lost load"] = get_lost_load(n.loads, n.generators)
    curtail = get_curtailment_per_type(n.generators, ["ccgt", "solar_tallmaxm", "wind_aerodyn"])
    lost_load_and_curtailment.loc[names[i]]["wind curt"] = curtail["wind_aerodyn"]
    lost_load_and_curtailment.loc[names[i]]["solar curt"] = curtail["solar_tallmaxm"]

    # Get costs
    gen_capex_costs = get_gen_capex_costs(n.generators, ["ccgt", "solar_tallmaxm", "wind_aerodyn"])
    tech_unit_capex_costs.loc[names[i]]["ccgt"] = gen_capex_costs["ccgt"]
    tech_unit_capex_costs.loc[names[i]]["solar"] = gen_capex_costs["solar_tallmaxm"]
    tech_unit_capex_costs.loc[names[i]]["wind"] = gen_capex_costs["wind_aerodyn"]
    tech_unit_capex_costs.loc[names[i]]["store"] = np.mean(n.storages.capital_cost.values)
    tech_unit_capex_costs.loc[names[i]]["trans"] = np.mean(n.lines.capital_cost.values)

    gen_opex_costs = get_gen_opex_costs(n.generators, ["ccgt", "solar_tallmaxm", "wind_aerodyn"])
    tech_unit_opex_costs.loc[names[i]]["ccgt"] = gen_opex_costs["ccgt"]
    tech_unit_opex_costs.loc[names[i]]["solar"] = gen_opex_costs["solar_tallmaxm"]
    tech_unit_opex_costs.loc[names[i]]["wind"] = gen_opex_costs["wind_aerodyn"]
    tech_unit_opex_costs.loc[names[i]]["store"] = np.mean(n.storages.marginal_cost.values)

    gen_total_capex_cost = get_total_gen_capex_costs(n.generators, ["ccgt", "solar_tallmaxm", "wind_aerodyn"])
    tech_capex_costs.loc[names[i]]["ccgt"] = gen_total_capex_cost["ccgt"]
    tech_capex_costs.loc[names[i]]["solar"] = gen_total_capex_cost["solar_tallmaxm"]
    tech_capex_costs.loc[names[i]]["wind"] = gen_total_capex_cost["wind_aerodyn"]
    tech_capex_costs.loc[names[i]]["trans"] = np.sum(n.lines.capital_cost*(n.lines.s_nom_opt-n.lines.s_nom)).item()
    tech_capex_costs.loc[names[i]]["store"] = np.sum(n.storages.capital_cost
                                                     * (n.storages.p_nom_opt-n.storages.p_nom)).item()
    tech_capex_costs.loc[names[i]]["total"] = tech_capex_costs.loc[names[i]]["ccgt"] + \
        tech_capex_costs.loc[names[i]]["solar"] + tech_capex_costs.loc[names[i]]["wind"] + \
        tech_capex_costs.loc[names[i]]["trans"] + tech_capex_costs.loc[names[i]]["store"]

    gen_total_opex_cost = get_total_gen_opex_costs(n.generators, ["ccgt", "solar_tallmaxm", "wind_aerodyn"])
    tech_opex_costs.loc[names[i]]["ccgt"] = gen_total_opex_cost["ccgt"]
    tech_opex_costs.loc[names[i]]["solar"] = gen_total_opex_cost["solar_tallmaxm"]
    tech_opex_costs.loc[names[i]]["wind"] = gen_total_opex_cost["wind_aerodyn"]
    tech_opex_costs.loc[names[i]]["trans"] = 0
    tech_opex_costs.loc[names[i]]["store"] = 0
    tech_opex_costs.loc[names[i]]["total"] = tech_opex_costs.loc[names[i]]["ccgt"] + \
        tech_opex_costs.loc[names[i]]["solar"] + tech_opex_costs.loc[names[i]]["wind"] + \
        tech_opex_costs.loc[names[i]]["trans"] + tech_opex_costs.loc[names[i]]["store"]

    lost_load_costs.loc[names[i]]["total"] = costs["lost_load"]*get_lost_load(n.loads, n.generators)

    total_costs.loc[names[i]]["total"] = lost_load_costs.loc[names[i]]["total"] + \
        tech_opex_costs.loc[names[i]]["total"] + tech_capex_costs.loc[names[i]]["total"]

    total_lines.loc[names[i]] = np.sum(n.lines.s_nom_opt*n.lines.length).item()/1000.
    new_lines.loc[names[i]] = np.sum((n.lines.s_nom_opt - n.lines.s_nom)*n.lines.length).item()/1000.
    """
    print("Length", get_length(n.lines).item())
    print("Max line capacity", get_max_trans(n.lines))
    print(np.sum(n.lines.length * n.lines.s_nom_opt))
    print(n.lines.where(n.lines.s_nom_opt == get_max_trans(n.lines), drop=True).id.item())
    print(n.lines.where(n.lines.s_nom_opt == get_max_trans(n.lines), drop=True).length.item())
    print(len(n.lines.id.values))
    """

print("Max cap\n", max_capacities)
print("Opt cap\n", opt_capacities)
print("Total gen\n", total_gen)
print("Lost load and curtailment\n", lost_load_and_curtailment)
# print("Capex costs per unit\n", tech_unit_capex_costs)
# print("Opex costs per unit\n", tech_unit_opex_costs)
print("Capex costs\n", tech_capex_costs)
print("Opex costs\n", tech_opex_costs)
print("Lost load cost\n", lost_load_costs)
print("Total cost\n", total_costs)
print("Total TWkm\n", total_lines)
print("New TWkm\n", new_lines)
