import plotly.graph_objs as go
from pypsa import Network
import numpy as np
import sys
import os


class PyPSAResultsPlotly:

    """
    This class allows to display results from a optimized PyPSA network.
    """

    def __init__(self, output_dir):

        self.current_test_number = test_number
        if self.current_test_number is None:
            self.current_test_number = sorted(os.listdir(output_dir))[-1]
        self.output_dir = output_dir
        self.net = Network()
        self.net.import_from_csv_folder(self.output_dir)
        self.tech_colors = {"All": "rgba(138,43,226,0.5)",  # purple
                            "nuclear": "rgba(255,140,0,0.5)",  # orange
                            "wind_onshore": "rgba(51,100,255,0.5)",  # middle-dark blue
                            "wind_offshore": "rgba(51,51,255,0.5)",  # dark blue
                            "wind_floating": "rgba(51,164,255,0.5)",  # middle blue
                            "pv_utility": "rgba(220,20,60,0.5)",  # red
                            "pv_residential": "rgba(220,20,20,0.5)",  # dark red
                            "ror": "rgba(255,153,255,0.5)",  # pink
                            "ccgt": "rgba(47,79,79,0.5)",  # grey
                            "ocgt": "rgba(105,105,105,0.5)",  # other grey
                            "Li-ion P": "rgba(102,255,178,0.5)",  # light green
                            "Li-ion E": "rgba(102,255,178,0.5)",  # light green
                            "phs": "rgba(0,153,76,0.5)",  # dark green
                            "sto": "rgba(51,51,255,0.5)",  # dark blue
                            "imports": "rgba(255,215,0,0.5)",  # yellow
                            }
        self.selected_types = sorted(list(set(self.net.generators.type.values)))

        """
        all_load_per_time = self.net.loads_t.p_set.to_numpy().sum(axis=1)
        all_generation = self.net.generators.p_nom_max.to_numpy()*self.net.generators_t.p_max_pu.to_numpy()
        all_generation_per_time = all_generation.sum(axis=1)
        print(sum([all_generation_per_time[i] >= all_load_per_time[i] for i in range(len(all_generation_per_time))])
              /len(all_generation_per_time))
        #for i in range(len(all_generation_per_time)):
        #    print(all_generation_per_time[i] - all_load_per_time[i], all_generation_per_time[i], all_load_per_time[i])
        """
    def show_topology(self):

        all_xs = np.concatenate((self.net.buses["x"].values, self.net.generators["x"].values,
                                 self.net.storage_units["x"].values))
        all_ys = np.concatenate((self.net.buses["y"].values, self.net.generators["y"].values,
                                 self.net.storage_units["y"].values))
        map_coords = [min(all_xs) - 5,
                      max(all_xs) + 5,
                      min(all_ys) - 2,
                      max(all_ys) + 2]

        fig = go.Figure(
            layout=go.Layout(
                showlegend=False,
                geo=dict(
                    showcountries=True,
                    scope='world',
                    lonaxis=dict(
                        showgrid=True,
                        gridwidth=1,
                        range=[map_coords[0], map_coords[1]],
                        dtick=5
                    ),
                    lataxis=dict(
                        showgrid=True,
                        gridwidth=1,
                        range=[map_coords[2], map_coords[3]],
                        dtick=5
                    )
                )
            )
        )

        # Adding lines to map
        # Get minimum s_nom_opt
        if len(self.net.lines) != 0:
            s_nom_opt_min = min(self.net.lines.s_nom_opt[self.net.lines.s_nom_opt > 0].values)
            for i, idx in enumerate(self.net.lines.index):
                bus0_id = self.net.lines.loc[idx, ("bus0",)]
                bus1_id = self.net.lines.loc[idx, ("bus1",)]
                bus0_x = self.net.buses.loc[bus0_id, ("x",)]
                bus0_y = self.net.buses.loc[bus0_id, ("y",)]
                bus1_x = self.net.buses.loc[bus1_id, ("x",)]
                bus1_y = self.net.buses.loc[bus1_id, ("y",)]
                color = 'rgba(0,0,255,0.8)'
                name = 'AC'
                s_nom_mul = self.net.lines.loc[idx, 's_nom_opt'] / s_nom_opt_min
                if self.net.lines.loc[idx, ("carrier",)] == "DC":
                    color = 'rgba(255,0,0,0.8)'
                    name = 'DC'

                fig.add_trace(go.Scattergeo(
                    mode='lines',
                    lon=[bus0_x, (bus0_x + bus1_x) / 2, bus1_x],
                    lat=[bus0_y, (bus0_y + bus1_y) / 2, bus1_y],
                    line=dict(
                        width=np.log(1 + s_nom_mul),
                        color=color),
                    text=[f"Name: {idx}<br>"
                          f"Init Capacity: {self.net.lines.loc[idx, 's_nom']}"
                          f"Opt Capacity: {self.net.lines.loc[idx, 's_nom_opt']}"]*3,
                    hoverinfo='text',
                    name=name
                ))

        # Adding links to map
        if len(self.net.links) != 0:
            # Get minimum p_nom_opt
            p_nom_opt_min = min(self.net.links.p_nom_opt[self.net.links.p_nom_opt > 0].values)
            for i, idx in enumerate(self.net.links.index):
                bus0_id = self.net.links.loc[idx, ("bus0",)]
                bus1_id = self.net.links.loc[idx, ("bus1",)]
                bus0_x = self.net.buses.loc[bus0_id, ("x",)]
                bus0_y = self.net.buses.loc[bus0_id, ("y",)]
                bus1_x = self.net.buses.loc[bus1_id, ("x",)]
                bus1_y = self.net.buses.loc[bus1_id, ("y",)]
                color = 'rgba(0,0,255,0.8)'
                p_nom_mul = self.net.links.loc[idx, 'p_nom_opt'] / p_nom_opt_min

                fig.add_trace(go.Scattergeo(
                    mode='lines',
                    lon=[bus0_x, (bus0_x + bus1_x) / 2, bus1_x],
                    lat=[bus0_y, (bus0_y + bus1_y) / 2, bus1_y],
                    line=dict(
                        width=np.log(1 + p_nom_mul),
                        color=color),
                    text=[f"Name: {idx}<br>"
                          f"Init Capacity: {self.net.links.loc[idx, 'p_nom']}<br>"
                          f"Opt Capacity: {self.net.links.loc[idx, 'p_nom_opt']}"]*3,
                    hoverinfo='text',
                ))

        # Add points to map
        p_noms = np.zeros((len(self.net.buses.index, )))
        color = self.tech_colors['All']
        if len(self.selected_types) == 1:
            color = self.tech_colors[self.selected_types[0]]
        colors = [color] * len(self.net.buses.index)
        for i, bus_id in enumerate(self.net.buses.index):
            total_gens = 0
            generators = self.net.generators[self.net.generators.bus == bus_id]

            # Keep only the reactors of the type we want to display
            for t in self.selected_types:
                generators_filter = generators[generators.type == t]
                p_noms[i] += np.sum(generators_filter["p_nom_opt"].values)
                total_gens += len(generators_filter["p_nom"].values)

            if total_gens == 0:
                # No allowed generation building
                colors[i] = 'grey'
            elif p_noms[i] == 0:
                colors[i] = 'black'

        p_nom_max = np.max(p_noms)
        if p_nom_max == 0:
            p_nom_max = 1  # Prevents cases where there is no installed capacity at all

        fig.add_trace(go.Scattergeo(
            mode="markers",
            lat=self.net.buses['y'].values,
            lon=self.net.buses['x'].values,
            text=[f"Bus: {self.net.buses.index[i]}<br>"
                  f"Total installed generation capacity: {p_noms[i]}" for i in range(len(self.net.buses))],
            hoverinfo='text',
            marker=dict(
                size=10 + 40 * np.log(1 + p_noms / p_nom_max),
                color=colors
            ),
            name='bus'
        ))

        return fig

    def show_generators(self, types, attribute):

        generators_idx = self.net.generators[self.net.generators.type.isin(types)].index

        print(self.net.generators.keys())

        all_xs = self.net.generators.loc[generators_idx, "x"].values
        all_ys = self.net.generators.loc[generators_idx, "y"].values

        map_coords = [min(all_xs) - 5, max(all_xs) + 5, min(all_ys) - 2, max(all_ys) + 2]

        fig = go.Figure(layout=go.Layout(
            showlegend=False,
            title=attribute + " for " + ",".join(types),
            geo=dict(
                showcountries=True,
                scope='world',
                lonaxis=dict(
                    showgrid=True,
                    gridwidth=1,
                    range=[map_coords[0], map_coords[1]],
                    dtick=5
                ),
                lataxis=dict(
                    showgrid=True,
                    gridwidth=1,
                    range=[map_coords[2], map_coords[3]],
                    dtick=5
                )
            )
        ))

        if attribute in self.net.generators_t.keys():
            filter_generators_idx = [idx for idx in generators_idx.values if idx in self.net.generators_t[attribute].keys()]
            values = self.net.generators_t[attribute][filter_generators_idx].mean()
            print(values)
        else:
            values = self.net.generators.loc[generators_idx, attribute].values

        colors = np.array([self.tech_colors[tech_type]
                           for tech_type in self.net.generators.loc[generators_idx].type.values])
        colors[[i for i in range(len(colors)) if values[i] == 0]] = 'black'

        max_value = np.max(values)
        if max_value == 0:
            max_value = 1  # Prevents cases where there is no installed capacity at all

        fig.add_trace(go.Scattergeo(
            mode="markers",
            lat=all_ys,
            lon=all_xs,
            text=[generators_idx[i] + " " + str(values[i]) for i in range(len(values))],
            hoverinfo='text',
            #name='generator',
            marker=dict(
                size=10 + 40 * np.log(1 + values / max_value),
                color=colors
            ),
        ))

        return fig

    def show_storage(self, types, attribute):

        print(self.net.storage_units)
        storage_idx = self.net.storage_units[self.net.storage_units.type.isin(types)].index
        print(storage_idx)

        print(self.net.storage_units.keys())

        all_xs = self.net.storage_units.loc[storage_idx, "x"].values
        all_ys = self.net.storage_units.loc[storage_idx, "y"].values

        map_coords = [min(all_xs) - 5, max(all_xs) + 5, min(all_ys) - 2, max(all_ys) + 2]

        fig = go.Figure(layout=go.Layout(
            showlegend=False,
            title=attribute + " for " + ",".join(types),
            geo=dict(
                showcountries=True,
                scope='world',
                lonaxis=dict(
                    showgrid=True,
                    gridwidth=1,
                    range=[map_coords[0], map_coords[1]],
                    dtick=5
                ),
                lataxis=dict(
                    showgrid=True,
                    gridwidth=1,
                    range=[map_coords[2], map_coords[3]],
                    dtick=5
                )
            )
        ))

        if attribute in self.net.storage_units_t.keys():
            print(self.net.storage_units_t[attribute])
            values = self.net.storage_units_t[attribute][storage_idx].mean()
            print(values)
        else:
            values = self.net.storage_units.loc[storage_idx, attribute].values

        colors = np.array([self.tech_colors[tech_type]
                           for tech_type in self.net.storage_units.loc[storage_idx].type.values])
        colors[[i for i in range(len(colors)) if values[i] == 0]] = 'black'

        max_value = np.max(values)
        if max_value == 0:
            max_value = 1  # Prevents cases where there is no installed capacity at all

        fig.add_trace(go.Scattergeo(
            mode="markers",
            lat=all_ys,
            lon=all_xs,
            text=[storage_idx[i] + " " + str(values[i]) for i in range(len(values))],
            hoverinfo='text',
            #name='generator',
            marker=dict(
                size=10 + 40 * np.log(1 + values / max_value),
                color=colors
            ),
        ))

        return fig


if __name__ == "__main__":

    assert (len(sys.argv) == 2) or (len(sys.argv) == 3), \
        "You need to provide one or two argument: output_dir (and test_number)"

    main_output_dir = sys.argv[1]
    test_number = sys.argv[2] if len(sys.argv) == 3 else None
    if test_number is None:
        test_number = sorted(os.listdir(main_output_dir))[-1]
    output_dir = main_output_dir + test_number + "/"
    print(output_dir)

    pprp = PyPSAResultsPlotly(output_dir)

    if 0:
        fig = pprp.show_topology()
        fig.write_html(output_dir + "topology.html", auto_open=True)
    if 1:
        types = ["wind_onshore", "wind_offshore", "pv_utility", "pv_residential", "ccgt"]
        attribute = "p_nom_opt"
        for tech_type in types:
            fig = pprp.show_generators([tech_type], attribute)
            fig.write_html(output_dir + attribute + "_for_" + "_".join([tech_type]) + '.html', auto_open=True)
    if 0:
        types = ["sto", "phs"]
        attribute = "p_nom_opt"
        for tech_type in types:
            fig = pprp.show_storage([tech_type], attribute)
            fig.write_html(output_dir + attribute + "_for_" + "_".join([tech_type]) + '.html', auto_open=True)
