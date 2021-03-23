from pyomo.environ import Constraint, NonNegativeReals
import pypsa


def store_links_constraint(network: pypsa.Network, ctd_ratio: float):

    model = network.model
    links = network.links

    links_to_bus = links[links.index.str.contains('to AC')].index
    links_from_bus = links[links.index.str.contains('AC to')].index

    def store_links_ratio_rule(model, discharge_link, charge_link):

        return model.link_p_nom[discharge_link]*ctd_ratio == model.link_p_nom[charge_link]

    model.store_links_ratio = Constraint(list(zip(links_to_bus, links_from_bus)), rule=store_links_ratio_rule)

