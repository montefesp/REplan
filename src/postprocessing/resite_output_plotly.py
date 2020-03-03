import plotly.graph_objs as go
from typing import Dict, List, Tuple
from src.resite.resite import Resite

class ResiteOutput:

    def __init__(self, resite: Resite):
        self.tech_colors = {"wind_onshore": "rgba(51,100,255,0.5)",  # middle-dark blue
                            "wind_offshore": "rgba(51,51,255,0.5)",  # dark blue
                            "wind_floating": "rgba(50,164,255,0.5)",  # middle blue
                            "pv_utility": "rgba(220,20,60,0.5)",  # red
                            "pv_residential": "rgba(255,153,255,0.5)",  # pink
                            }
        self.resite = resite

    def show_points(self):
        """
        This function displays the list of geographical points (lon, lat) associated to a series of technology.

        Parameters
        ----------
        tech_points_dict: Dict[str, List[Tuple[float, float]]]
            Dictionary associating a list of points (lon, lat) to each technology

        """

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

            # Original points
            fig.add_trace(go.Scattergeo(
                mode="markers",
                lat=ys,
                lon=xs,
                # text=self.net.buses.index,
                hoverinfo='text',
                marker=dict(
                    color='black',
                    size=5
                ),
                name='bus'
            ))

            if tech in self.resite.selected_tech_points_dict:

                selected_points = self.resite.selected_tech_points_dict[tech]

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
                        size=10
                    ),
                    name='bus'
                ))

            fig.write_html(f'{tech}.html', auto_open=True)
