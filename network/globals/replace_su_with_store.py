import pypsa

import logging
logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(asctime)s - %(message)s")
logger = logging.getLogger()


def replace_su_closed_loop(network: pypsa.Network, su_to_replace: str):

    su = network.storage_units.loc[su_to_replace]

    su_short_name = su_to_replace.split(' ')[-1]
    bus_name = f"{su['bus']} {su_short_name}"
    link_1_name = f"{su['bus']} link {su_short_name} to AC"
    link_2_name = f"{su['bus']} link AC to {su_short_name}"
    store_name = f"{bus_name} store"

    # add bus
    network.add("Bus", bus_name)

    # add discharge link
    network.add("Link",
                link_1_name,
                bus0=bus_name,
                bus1=su["bus"],
                capital_cost=su["capital_cost"],
                marginal_cost=su["marginal_cost"],
                p_nom_extendable=su["p_nom_extendable"],
                p_nom=su["p_nom"]/su["efficiency_dispatch"],
                p_nom_min=su["p_nom_min"]/su["efficiency_dispatch"],
                p_nom_max=su["p_nom_max"]/su["efficiency_dispatch"],
                p_max_pu=su["p_max_pu"],
                efficiency=su["efficiency_dispatch"])

    # add charge link
    network.add("Link",
                link_2_name,
                bus1=bus_name,
                bus0=su["bus"],
                p_nom=su["p_nom"],
                p_nom_extendable=su["p_nom_extendable"],
                p_nom_min=su["p_nom_min"],
                p_nom_max=su["p_nom_max"],
                p_max_pu=-su["p_min_pu"],
                efficiency=su["efficiency_store"])

    # add store
    network.add("Store",
                store_name,
                bus=bus_name,
                capital_cost=su["capital_cost_e"],
                marginal_cost=su["marginal_cost_e"],
                e_nom=su["p_nom"]*su["max_hours"],
                e_nom_min=su["p_nom_min"]/su["efficiency_dispatch"]*su["max_hours"],
                e_nom_max=su["p_nom_max"]/su["efficiency_dispatch"]*su["max_hours"],
                e_nom_extendable=su["p_nom_extendable"],
                e_max_pu=1.,
                e_min_pu=0.,
                standing_loss=su["standing_loss"],
                e_cyclic=su['cyclic_state_of_charge'])

    network.remove("StorageUnit", su_to_replace)
