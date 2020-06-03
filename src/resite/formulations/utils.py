from typing import Tuple, Dict, List
from os.path import join

import pandas as pd


def write_lp_file(model, modelling: str, output_folder: str):

    accepted_modelling = ["docplex", "gurobipy", "pyomo"]
    assert modelling in accepted_modelling, f"Error: This function does not work with modelling language {modelling}."

    if modelling == "docplex":
        model.pprint(out=join(output_folder, 'model_resite_docplex.lp'))
    if modelling == "gurobipy":
        model.write(join(output_folder, 'model_resite_gurobipy.lp'))
    elif modelling == "pyomo":
        from pyomo.opt import ProblemFormat
        model.write(filename=join(output_folder, 'model_resite_pyomo.lp'),
                    format=ProblemFormat.cpxlp,
                    io_options={'symbolic_solver_labels': True})


def solve_model(resite) -> float:
    """Solve the model and return the objective value."""

    if resite.modelling == "docplex":
        resite.instance.print_information()
        resite.instance.context.solver.log_output = True
        resite.instance.solve()
        objective = resite.instance.objective_value
        print(f"Objective value: {objective}")
    elif resite.modelling == "gurobipy":
        resite.instance.optimize()
        objective = resite.obj.getValue()
    else:  # resite.modelling == "pyomo":
        from pyomo.opt import SolverFactory
        from pyomo.environ import value
        opt = SolverFactory('cbc')
        results = opt.solve(resite.instance, tee=True, keepfiles=False, report_timing=False)
        resite.results = results
        objective = value(resite.instance.objective)

    resite.objective = objective
    return objective


# TODO: should this directly be in the formulation file? - probably
def retrieve_solution(resite) -> Tuple[Dict[str, List[Tuple[float, float]]], pd.Series]:

    """
    Get the solution of the optimization.

    Returns
    -------
    selected_tech_points_dict: Dict[str, List[Tuple[float, float]]]
        Lists of points for each technology used in the model
    optimal_cap_ds: pd.Series
        Gives for each pair of technology-location the optimal capacity obtained via the optimization

    """

    optimal_cap_ds = pd.Series(index=pd.MultiIndex.from_tuples(resite.tech_points_tuples))
    selected_tech_points_dict = {tech: [] for tech in resite.technologies}

    tech_points_tuples = [(tech, coord[0], coord[1]) for tech, coord in resite.tech_points_tuples]
    for tech, lon, lat in tech_points_tuples:

        if resite.modelling == "docplex":
            y_value = resite.instance.y[tech, lon, lat].solution_value
        elif resite.modelling == "gurobipy":
            y_value = resite.y[tech, lon, lat].X
        else:  # modelling == "pyomo":
            y_value = resite.instance.y[tech, (lon, lat)].value

        optimal_cap_ds[tech, (lon, lat)] = y_value*resite.cap_potential_ds[tech, (lon, lat)]
        if y_value > 0.:
            selected_tech_points_dict[tech] += [(lon, lat)]

    # Remove tech for which no points was selected
    selected_tech_points_dict = {k: v for k, v in selected_tech_points_dict.items() if len(v) > 0}

    return selected_tech_points_dict, optimal_cap_ds
