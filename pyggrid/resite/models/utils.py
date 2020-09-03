from os.path import join

import pandas as pd

import logging

def write_lp_file(model, modelling: str, output_folder: str):
    """Save LP file of the model."""

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


# TODO: should build and solve become one single function?
def solve_model(resite) -> None:
    """Solve the model and retrieve the solution."""

    if resite.modelling == "docplex":
        resite.instance.print_information()
        resite.instance.context.solver.log_output = True
        resite.instance.solve()
        objective = resite.instance.objective_value
        print(f"Objective value: {objective}")
    elif resite.modelling == "gurobipy":
        resite.instance.optimize()
        # from gurobipy import GRB
        # status = resite.instance.status
        # if status == GRB.INFEASIBLE
        objective = resite.obj.getValue()
    elif resite.modelling == "pyomo":
        from pyomo.opt import SolverFactory
        from pyomo.environ import value
        opt = SolverFactory('gurobi')
        results = opt.solve(resite.instance, tee=True, keepfiles=False, report_timing=False)
        resite.results = results
        objective = value(resite.instance.objective)
        # if str(results.solver.termination_condition) != "infeasible"):
    else:
        logging.info("No implementation has been written for solving with this modelling language.")
        return

    resite.objective = objective
    retrieve_solution(resite)


def retrieve_solution(resite) -> None:
    """Get the solution of the optimization."""

    # Portion of potential capacity selected by the model for all sites (selected and non-selected).
    y_ds = pd.Series(index=pd.MultiIndex.from_tuples(resite.tech_points_tuples))
    # Selected sites stored as dictionary indexed on technology.
    sel_tech_points_dict = {tech: [] for tech in resite.technologies}

    for tech, lon, lat in resite.tech_points_tuples:

        if resite.modelling == "docplex":
            y_value = resite.instance.y[tech, lon, lat].solution_value
        elif resite.modelling == "gurobipy":
            y_value = resite.y[tech, lon, lat].X
        else:  # modelling == "pyomo":
            y_value = resite.instance.y[tech, lon, lat].value

        y_ds[tech, lon, lat] = y_value
        if y_value > 0.:
            sel_tech_points_dict[tech] += [(lon, lat)]

    resite.y_ds = y_ds
    # Remove tech for which no points was selected
    resite.sel_tech_points_dict = {k: v for k, v in sel_tech_points_dict.items() if len(v) > 0}
