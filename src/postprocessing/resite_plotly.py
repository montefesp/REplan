import sys
import os
from os.path import join
import pickle

import plotly.graph_objs as go

from src.resite.resite import Resite


class ResitePlotly:

    def __init__(self, resite: Resite):
        self.tech_colors = {"wind_onshore": "rgba(51,100,255,0.5)",  # middle-dark blue
                            "wind_offshore": "rgba(51,51,255,0.5)",  # dark blue
                            "wind_floating": "rgba(50,164,255,0.5)",  # middle blue
                            "pv_utility": "rgba(220,20,60,0.5)",  # red
                            "pv_residential": "rgba(255,153,255,0.5)",  # pink
                            }
        self.resite = resite
        self.output_dir = resite.output_folder

    def show_points(self, color_variable: str, auto_open: bool):
        """This function displays the list of geographical points (lon, lat) associated to a series of technology."""
        for tech, initial_points in self.resite.tech_points_dict.items():

            xs = [x for x, _ in initial_points]
            ys = [y for _, y in initial_points]
            minx = min(xs) - 2
            maxx = max(xs) + 2
            miny = min(ys) - 2
            maxy = max(ys) + 2

            fig = go.Figure(layout=go.Layout(
                showlegend=True,
                legend_orientation="h",
                title=f"Selected points for {tech}<br>"
                      f"Formulation: {self.resite.formulation}<br>"
                      f"Technologies: {self.resite.technologies}<br>"
                      f"Deployment vector: {self.resite.deployment_vector}",
                geo=dict(
                    showcountries=True,
                    scope='world',
                    lonaxis=dict(
                        showgrid=True,
                        gridwidth=1,
                        range=[minx, maxx],
                        dtick=5
                    ),
                    lataxis=dict(
                        showgrid=True,
                        gridwidth=1,
                        range=[miny, maxy],
                        dtick=5
                    )
                )
            ))

            cap_potential_ds = self.resite.cap_potential_ds[tech]
            avg_cap_factor = self.resite.cap_factor_df[tech].mean(axis=0)
            existing_capacity_ds = self.resite.existing_capacity_ds[tech]
            optimal_capacity_ds = self.resite.optimal_capacity_ds[tech]

            # Original points
            fig.add_trace(
                go.Scattergeo(
                    mode="markers",
                    lat=ys,
                    lon=xs,
                    text=[f"Capacity potential: {cap_potential_ds[index]:.4f}<br>"
                          f"Initial capacity: {existing_capacity_ds[index]:.4f}<br>"
                          f"Optimal capacity: {optimal_capacity_ds[index]:.4f}<br>"
                          f"Average cap factor: {avg_cap_factor[index]:.4f}" for index in cap_potential_ds.index],
                    hoverinfo='text',
                    marker=dict(
                        color='black',
                        size=5
                    ),
                    name="Available points"))

            if tech in self.resite.selected_tech_points_dict:

                colorscale = 'Blues'
                if tech.split("_")[0] == 'pv':
                    colorscale = 'Reds'

                selected_points = self.resite.selected_tech_points_dict[tech]
                if color_variable == 'optimal_capacity':
                    color_values = optimal_capacity_ds[selected_points]
                    colorbar_title = "Optimal capacity"
                elif color_variable == 'percentage_of_potential':
                    color_values = optimal_capacity_ds[selected_points]/cap_potential_ds[selected_points]
                    colorbar_title = "Percentage of potential installed"

                # Get points with existing capacity
                pos_existing_capacity_ds = existing_capacity_ds[existing_capacity_ds > 0]
                points_with_ex_cap = [point for point in selected_points if point in pos_existing_capacity_ds.index]
                if len(points_with_ex_cap):
                    fig.add_trace(go.Scattergeo(
                        mode="markers",
                        lat=[y for _, y in points_with_ex_cap],
                        lon=[x for x, _ in points_with_ex_cap],
                        name="Selected points (with existing capacity)",
                        hoverinfo='text',
                        marker=dict(
                            size=10,
                            opacity=0.8,
                            reversescale=True,
                            autocolorscale=False,
                            symbol='x',
                            line=dict(
                                width=1,
                                color='rgba(102, 102, 102)'
                            ),
                            colorscale=colorscale,
                            cmin=0,
                            color=color_values[points_with_ex_cap],
                            cmax=color_values.max(),
                            colorbar_title=colorbar_title
                    )))

                # Plotting points without existing cap
                points_without_ex_cap = [point for point in selected_points
                                         if point not in pos_existing_capacity_ds.index]
                if len(points_without_ex_cap):
                    fig.add_trace(go.Scattergeo(
                        mode="markers",
                        lat=[y for _, y in points_without_ex_cap],
                        lon=[x for x, _ in points_without_ex_cap],
                        name="Selected points (without existing capacity)",
                        hoverinfo='text',
                        marker=dict(
                            size=10,
                            opacity=0.8,
                            reversescale=True,
                            autocolorscale=False,
                            symbol='circle',
                            line=dict(
                                width=1,
                                color='rgba(102, 102, 102)'
                            ),
                            colorscale=colorscale,
                            cmin=0,
                            color=color_values[points_without_ex_cap],
                            cmax=color_values.max(),
                            colorbar_title=colorbar_title
                    )))

            fig.write_html(join(self.output_dir, f'{color_variable}_{tech}.html'), auto_open=auto_open)

    def analyse_feasibility(self, auto_open=True):

        if self.resite.formulation != "meet_RES_targets_hourly":
            print(f"Error: This function is only implemented for formulation "
                  f"meet_RES_targets_hourly not for {self.resite.formulation}")

        # Compute number of time-steps for which the constraint was not feasible
        print(sum(self.resite.generation_potential_df.sum(axis=1) >
                  self.resite.deployment_vector[0]*self.resite.load_df.sum(axis=1)))

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=self.resite.timestamps,
                                 y=self.resite.generation_potential_df.sum(axis=1),
                                 name="Total Generation Potential (GWh)"))
        fig.add_trace(go.Scatter(x=self.resite.timestamps,
                                 y=self.resite.load_df.sum(axis=1)*0.3, # *self.resite.deployment_vector[0],
                                 name="Portion of the load to be served (GWh)"))

        fig.write_html(join(self.output_dir, 'infeasibility_study.html'), auto_open=auto_open)


if __name__ == "__main__":

    assert (len(sys.argv) == 2) or (len(sys.argv) == 3), \
        "You need to provide one or two argument: output_dir (and test_number)"

    main_output_dir = sys.argv[1]
    test_number = sys.argv[2] if len(sys.argv) == 3 else None
    if test_number is None:
        test_number = sorted(os.listdir(main_output_dir))[-1]
    output_dir = main_output_dir + test_number + "/"
    print(output_dir)

    resite = pickle.load(open(output_dir + "resite_model.p", 'rb'))
    print(f"Region: {resite.regions}")
    ro = ResitePlotly(resite)

    ro.get_total_capacity()
    exit()

    if resite.modelling != "pyomo" or \
            (resite.modelling == "pyomo" and str(resite.results.solver.termination_condition) != "infeasible"):
        ro.show_points("percentage_of_potential", auto_open=True)
        ro.show_points("optimal_capacity", auto_open=True)

    # Works even if infeasible
    ro.analyse_feasibility(auto_open=True)

