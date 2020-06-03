from typing import List, Dict


def build_model(resite, modelling: str, params: Dict):
    """
    Model build-up.

    Parameters:
    ------------
    modelling: str
        Name of the modelling language to use.
    params: List[float]
        List of parameters needed by the formulation
    """
    accepted_modelling = ["pyomo"]
    assert modelling in accepted_modelling, f"Error: This formulation was not coded with modelling language {modelling}"

    assert 'nb_sites_per_region' in params and len(params['nb_sites_per_region']) == len(resite.regions), \
        "Error: This formulation requires a vector of required number of sites per region."

    build_model_ = globals()[f"build_model_{modelling}"]
    build_model_(resite, params['nb_sites_per_region'])


def build_model_pyomo(resite, nb_sites_per_region: List[float]):
    """Model build-up with pyomo"""

    from pyomo.environ import ConcreteModel, Binary, Var
    from src.resite.models.pyomo_aux import limit_number_of_sites_per_region, maximize_production

    tech_points_tuples = [(tech, coord[0], coord[1]) for tech, coord in resite.tech_points_tuples]
    regions = resite.regions

    model = ConcreteModel()

    # - Parameters - #
    nb_sites_per_region_dict = dict(zip(regions, nb_sites_per_region))

    # - Variables - #
    # Variables for the portion of capacity at each location for each technology
    model.y = Var(tech_points_tuples, within=Binary)

    # - Constraints - #
    model.policy_target = limit_number_of_sites_per_region(model, regions,
                                                           resite.region_tech_points_dict, nb_sites_per_region_dict)
    # - Objective - #
    model.objective = maximize_production(model, resite.cap_factor_df, tech_points_tuples)

    resite.instance = model