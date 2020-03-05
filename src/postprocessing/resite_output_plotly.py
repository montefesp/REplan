import plotly.graph_objs as go
from typing import Dict, List, Tuple
from src.resite.resite import Resite
import sys
import os
from os.path import join
import pickle


class ResiteOutput:

    def __init__(self, resite: Resite):
        self.tech_colors = {"wind_onshore": "rgba(51,100,255,0.5)",  # middle-dark blue
                            "wind_offshore": "rgba(51,51,255,0.5)",  # dark blue
                            "wind_floating": "rgba(50,164,255,0.5)",  # middle blue
                            "pv_utility": "rgba(220,20,60,0.5)",  # red
                            "pv_residential": "rgba(255,153,255,0.5)",  # pink
                            }
        self.resite = resite
        self.output_dir = resite.output_folder

    def show_points(self):
        """This function displays the list of geographical points (lon, lat) associated to a series of technology."""

        for tech, initial_points in self.resite.tech_points_dict.items():

            xs = [x for x, _ in initial_points]
            ys = [y for _, y in initial_points]
            minx = min(xs) - 2
            maxx = max(xs) + 2
            miny = min(ys) - 2
            maxy = max(ys) + 2

            fig = go.Figure(layout=go.Layout(
                showlegend=False,
                title=f"Selected points for {tech}",
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
            fig.add_trace(go.Scattergeo(
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
                name='bus'
            ))

            if tech in self.resite.selected_tech_points_dict:

                selected_points = self.resite.selected_tech_points_dict[tech]

                # Get points with existing capacity
                points_with_ex_cap = existing_capacity_ds[existing_capacity_ds > 0]
                symbols = ['x' if point in points_with_ex_cap.index else 'circle' for point in selected_points]

                xs = [x for x, _ in selected_points]
                ys = [y for _, y in selected_points]

                fig.add_trace(go.Scattergeo(
                    mode="markers",
                    lat=ys,
                    lon=xs,
                    # text=self.net.buses.index,
                    hoverinfo='text',
                    marker=dict(
                        color=self.tech_colors[tech],
                        size=10,
                        symbol=symbols
                    ),
                    name='bus'
                ))

            fig.write_html(join(self.output_dir, f'{tech}.html'), auto_open=True)


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
    resite_output = ResiteOutput(resite)
    resite_output.show_points()
