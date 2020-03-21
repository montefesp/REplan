from typing import List
import plotly.graph_objs as go
from pypsa import Network
import numpy as np
import pandas as pd
import sys
import os
from geojson import Polygon, FeatureCollection, MultiPolygon, Feature
from shapely.ops import cascaded_union
from shapely.geometry import Polygon as sPolygon, MultiPolygon as sMultiPolygon, Point
import shapely.wkt

from src.data.geographics.manager import get_offshore_shapes
from src.data.res_potential.manager import get_capacity_potential_for_regions

def get_map_layout(title: str, map_coords: List[float], showcountries=True):

    assert len(map_coords) == 4, "Error: map_coords must be of length 4"

    return go.Figure(
        layout=go.Layout(
            showlegend=False,
            title=dict(text=title, xanchor="center", xref='paper', x=0.5, font=dict(size=30)),
            geo=dict(
                showcountries=showcountries,
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
        self.opacity = 1.0
        self.tech_colors = {"All": f"rgba(138,43,226,{self.opacity})",  # purple
                            "nuclear": f"rgba(255,140,0,{self.opacity})",  # orange
                            "wind_onshore": f"rgba(150,0,200,{self.opacity})",  # purple
                            "wind_offshore": f"rgba(100,0,150,{self.opacity})",  # purple
                            "wind_floating": f"rgba(102,255,178,{self.opacity})",  # middle blue
                            "pv_utility": f"rgba(220,20,60,{self.opacity})",  # red
                            "pv_residential": f"rgba(220,20,20,{self.opacity})",  # dark red
                            "ror": f"rgba(100,100,255,{self.opacity})",  # light blue
                            "ccgt": f"rgba(47,79,79,{self.opacity})",  # grey
                            "ocgt": f"rgba(105,105,105,{self.opacity})",  # other grey
                            "Li-ion P": f"rgba(102,255,178,{self.opacity})",  # light green
                            "Li-ion E": f"rgba(102,255,178,{self.opacity})",  # light green
                            "phs": f"rgba(0,153,76,{self.opacity})",  # dark green
                            "sto": f"rgba(51,51,230,{self.opacity})",  # dark blue
                            "load": f"rgba(20,20,20,{self.opacity})",  # dark grey
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

        fig = get_map_layout("Topology", [min(all_xs) - 5, max(all_xs) + 5, min(all_ys) - 2, max(all_ys) + 2])

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
                        width=np.log(1 + p_nom_mul)/2,
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

    def show_topology_heatmap(self):

        all_xs = np.concatenate((self.net.buses["x"].values, self.net.generators["x"].values,
                                 self.net.storage_units["x"].values))
        all_ys = np.concatenate((self.net.buses["y"].values, self.net.generators["y"].values,
                                 self.net.storage_units["y"].values))
        fig = get_map_layout("Topology", [min(all_xs) - 5, max(all_xs) + 5, min(all_ys) - 2, max(all_ys) + 2])

        if 0:
            avg_uses = self.net.links_t.p0.abs().mean(axis=0)/self.net.links.p_nom_opt
            cmax = 1
            colorbar_title = "Mean (Power/Capacity) over time"
        if 0:
            limit_perc = 0.8
            avg_uses = (self.net.links_t.p0.abs() > 0.8*self.net.links.p_nom_opt).mean(axis=0)
            cmax = 1
            colorbar_title = f"Percentage of time where power<br>is above {limit_perc} of capacity"
        if 0:
            limit_perc = 0.8
            avg_uses = (self.net.links_t.p0.abs() > 0.8*self.net.links.p_nom_opt).sum(axis=0)
            cmax = 8760.
            colorbar_title = f"Number of hours where power<br>is above {limit_perc} of capacity"
        if 1:
            avg_uses = self.net.links.p_nom_opt - self.net.links.p_nom
            cmax = max(avg_uses.values)
            colorbar_title = "Increase in Capacity"

        # Adding links to map
        if len(self.net.links) != 0:
            # Get max p_nom_opt
            for i, idx in enumerate(self.net.links.index):
                bus0_id = self.net.links.loc[idx, "bus0"]
                bus1_id = self.net.links.loc[idx, "bus1"]
                bus0_x = self.net.buses.loc[bus0_id, "x"]
                bus0_y = self.net.buses.loc[bus0_id, "y"]
                bus1_x = self.net.buses.loc[bus1_id, "x"]
                bus1_y = self.net.buses.loc[bus1_id, "y"]

                avg_use = avg_uses[idx]
                color = f'rgba(0,0,255,{round(avg_use/cmax, 2)})'

                fig.add_trace(go.Scattergeo(
                    mode='lines+markers',
                    lon=[bus0_x, (bus0_x + bus1_x) / 2, bus1_x],
                    lat=[bus0_y, (bus0_y + bus1_y) / 2, bus1_y],
                    line=dict(
                        width=1,  # np.log(1 + p_nom_mul)/2,
                        color='black'),
                    marker=dict(
                        size=[0, 10, 0],
                        opacity=0.8,
                        reversescale=False,
                        autocolorscale=False,
                        symbol='circle',
                        line=dict(
                            width=1,
                            color='rgba(102, 102, 102)'
                        ),
                        colorscale='bluered',
                        cmin=0,
                        color=[avg_use]*3,
                        cmax=cmax,
                        colorbar_title=colorbar_title
                    ),
                    text=["", f"{avg_use:.2f}", ""],
                    hoverinfo="text"
                ))

        return fig

    def show_generators(self, types, attribute):

        # Get generators of given type
        generators_idx = self.net.generators[self.net.generators.type.isin(types)].index
        if len(generators_idx) == 0:
            print(f"Warning: There is no generator of type {types} in the model.")
            raise ValueError
        generators = self.net.generators.loc[generators_idx]

        # Get boundaries and general layout
        all_xs = generators.x.values
        all_ys = generators.y.values
        map_coords = [min(all_xs) - 5, max(all_xs) + 5, min(all_ys) - 2, max(all_ys) + 2]
        title = attribute + " for " + ",".join(types)
        fig = get_map_layout(title, map_coords)

        #
        if attribute in self.net.generators_t.keys():
            filter_generators_idx = [idx for idx in generators_idx.values if idx in self.net.generators_t[attribute].keys()]
            values = self.net.generators_t[attribute][filter_generators_idx].mean()
        else:
            values = generators[attribute].values

        # Get colors of points and size
        colors = np.array([self.tech_colors[tech_type] for tech_type in generators.type.values])
        colors[[i for i in range(len(colors)) if values[i] == 0]] = 'black'
        max_value = np.max(values)
        if max_value == 0:
            max_value = 1  # Prevents cases where there is no installed capacity at all
        sizes = 10 + 40 * np.log(1 + values / max_value)

        fig.add_trace(go.Scattergeo(
            mode="markers",
            lat=all_ys,
            lon=all_xs,
            text=[f"Name: {idx}<br>"
                  f"Init cap: {generators.loc[idx].p_nom}<br>"
                  f"Opt cap: {generators.loc[idx].p_nom_opt}" for idx in generators_idx],
            hoverinfo='text',
            #name='generator',
            marker=dict(
                size=sizes,
                color=colors
            ),
        ))

        return fig

    def show_storage(self, types, attribute):

        storage_idx = self.net.storage_units[self.net.storage_units.type.isin(types)].index
        if len(storage_idx) == 0:
            print(f"Warning: There is no generator of type {types} in the model.")
            raise ValueError

        all_xs = self.net.storage_units.loc[storage_idx, "x"].values
        all_ys = self.net.storage_units.loc[storage_idx, "y"].values
        map_coords = [min(all_xs) - 5, max(all_xs) + 5, min(all_ys) - 2, max(all_ys) + 2]
        title = attribute + " for " + ",".join(types)
        fig = get_map_layout(title, map_coords)

        if attribute in self.net.storage_units_t.keys():
            values = self.net.storage_units_t[attribute][storage_idx].mean()
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

    def show_bus_marginal_price_choropleth(self, case):

        # buses_onshore_index = self.net.buses.index
        buses_onshore_index = self.net.buses[self.net.buses.onshore].index

        if case == "peak":
            # Get peak load time-step
            peak_load_ts = self.net.loads_t.p_set.sum(axis=1).idxmax()

            # Get marginal cost for every bus
            marginal_price = (self.net.buses_t.marginal_price.loc[peak_load_ts, buses_onshore_index]*1e3).round(2)

            title = f"Marginal price at {peak_load_ts}"

        elif case == "avg":
            marginal_price = (self.net.buses_t.marginal_price[buses_onshore_index].mean(axis=0)*1e3).round(2)
            title = f"Average marginal price"
        elif case == "median":
            marginal_price = (self.net.buses_t.marginal_price[buses_onshore_index].median(axis=0)*1e3).round(2)
            title = f"Median marginal price"
        elif case == "max":
            marginal_price = (self.net.buses_t.marginal_price[buses_onshore_index].max(axis=0)*1e3).round(2)
            title = f"Max marginal price"
        elif case == "min":
            marginal_price = (self.net.buses_t.marginal_price[buses_onshore_index].min(axis=0)*1e3).round(2)
            title = f"Min marginal price"

        all_xs = self.net.buses.x.values
        all_ys = self.net.buses.y.values

        fig = get_map_layout(title, [min(all_xs) - 5, max(all_xs) + 5, min(all_ys) - 2, max(all_ys) + 2], False)

        feature_collection = []
        onshore_buses = self.net.buses.loc[buses_onshore_index]
        for idx in buses_onshore_index:
            region = shapely.wkt.loads(onshore_buses.region[idx])
            if isinstance(region, sPolygon):
                feature_collection += [Feature(geometry=Polygon([list(region.exterior.coords)]), id=idx)]
            else:
                feature_collection += [Feature(geometry=MultiPolygon([(list(poly.exterior.coords), ) for poly in region]), id=idx)]
        feature_collection = FeatureCollection(feature_collection)

        fig.add_trace(go.Choropleth(
            locations=buses_onshore_index,
            geojson=feature_collection,
            z=marginal_price,
            text=buses_onshore_index,
            colorscale='bluered',
            autocolorscale=False,
            reversescale=True,
            marker_line_color='black',
            marker_line_width=0.5,
            colorbar_ticksuffix=' €/MWh',
            colorbar_title='Marginal Price'
        ))

        return fig

    def get_map_divided_by_region(self, regions_dict, strategy='siting'):

        all_xs = self.net.buses.x.values
        all_ys = self.net.buses.y.values

        minx = min(all_xs) - 5
        maxx = max(all_xs) + 5
        miny = min(all_ys) - 2
        maxy = max(all_ys) + 2

        fig = get_map_layout("", [minx, maxx, miny, maxy], False)

        from functools import reduce
        all_countries = sorted(reduce(lambda x, y: x + y, list(regions_dict.values())))
        onshore_shape_union = \
            cascaded_union([shapely.wkt.loads(region) for region in self.net.buses[self.net.buses.onshore].region.values])
        offshore_shapes = get_offshore_shapes(all_countries, onshore_shape_union, filterremote=True)
        offshore_shapes.index = ['UK' if idx == "GB" else idx for idx in offshore_shapes.index]

        if strategy == "bus":
            # Compute capacity potential per eez
            tech_regions_dict = {"wind_offshore": offshore_shapes["geometry"].values}
            wind_capacity_potential_per_country = get_capacity_potential_for_regions(tech_regions_dict)['wind_offshore']
            wind_capacity_potential_per_country.index = offshore_shapes.index

            # Compute generation per offshore bus
            offshore_buses_index = self.net.buses[self.net.buses.onshore == False].index
            total_generation_per_bus = pd.Series(index=offshore_buses_index)
            total_max_capacity_per_bus = pd.Series(index=offshore_buses_index)
            for idx in offshore_buses_index:
                offshore_generators_index = self.net.generators[self.net.generators.bus == idx].index
                total_generation_per_bus[idx] = self.net.generators_t.p[offshore_generators_index].values.sum()
                total_max_capacity_per_bus[idx] = self.net.generators.loc[offshore_generators_index, 'p_nom_max'].values.sum()
            print(total_max_capacity_per_bus)

            offshore_bus_region_shapes = self.net.buses.loc[offshore_buses_index].region

        feature_collection = []
        for idx, regions in regions_dict.items():

            print(regions)

            # Get buses in region
            buses_index = self.net.buses.loc[[idx for idx in self.net.buses.index if idx[2:4] in regions]].index

            # Agglomerate regions together
            region_shape = \
                cascaded_union([shapely.wkt.loads(self.net.buses.loc[bus_id].region) for bus_id in buses_index])

            centroid = region_shape.centroid

            if isinstance(region_shape, sPolygon):
                feature_collection += [Feature(geometry=Polygon([list(region_shape.exterior.coords)]), id=idx)]
            else:
                feature_collection += \
                    [Feature(geometry=MultiPolygon([(list(poly.exterior.coords), ) for poly in region_shape]), id=idx)]

            # Get all generators for those buses
            generators = self.net.generators[self.net.generators.bus.isin(buses_index)]
            all_cap = dict.fromkeys(sorted(set(generators.type)))
            for key in all_cap:
                generators_type_index = generators[generators.type == key].index
                all_cap[key] = np.sum(self.net.generators_t.p[generators_type_index].values)

            # Add STO output
            storage_units = self.net.storage_units[self.net.storage_units.bus.isin(buses_index)]
            stos = storage_units[storage_units.type == "sto"]
            all_cap['sto'] = np.sum(self.net.storage_units_t.p[stos.index].values)

            # Add wind_offshore
            offshore_regions = [r for r in regions if r in offshore_shapes.index]
            eez_region_shape = cascaded_union(offshore_shapes.loc[offshore_regions]["geometry"])
            offshore_generators = self.net.generators[self.net.generators.type == 'wind_offshore']
            if strategy == 'siting':
                offshore_generators_in_region = \
                    offshore_generators[["x", "y"]].apply(lambda x: eez_region_shape.contains(Point(x[0], x[1])), axis=1)
                offshore_generators_in_region_index = offshore_generators[offshore_generators_in_region].index
                if len(offshore_generators_in_region_index) != 0:
                    all_cap['wind_offshore'] = np.sum(self.net.generators_t.p[offshore_generators_in_region_index].values)

            elif strategy == 'bus':
                wind_capacity = wind_capacity_potential_per_country[offshore_regions].sum()
                # Compute intersection with all offshore shapes
                all_cap['wind_offshore'] = 0
                for off_idx, off_region_shape in offshore_bus_region_shapes.items():
                    off_region_shape = shapely.wkt.loads(off_region_shape)
                    intersection = off_region_shape.intersection(eez_region_shape)
                    prop_cap_received_by_bus = (intersection.area/eez_region_shape.area)*wind_capacity
                    all_cap['wind_offshore'] += (prop_cap_received_by_bus/total_max_capacity_per_bus[off_idx])*total_generation_per_bus[off_idx]
                print(all_cap['wind_offshore'])

            x = (centroid.x - minx) / (maxx - minx)
            y = (centroid.y - miny) / (maxy - miny)

            title = idx
            if ' ' in title:
                title = title.split(" ")[0] + "<br>" + title.split(" ")[1]

            # Sort values
            sorted_keys = sorted(list(all_cap.keys()))
            sorted_values = [all_cap[key] for key in sorted_keys]

            fig.add_trace(go.Pie(
                values=sorted_values,
                labels=sorted_keys,
                hole=0.4,
                text=[""]*len(all_cap.keys()),
                textposition="none",
                scalegroup='one',
                domain=dict(x=[max(x - 0.14, 0), min(x + 0.14, 1.0)],
                            y=[max(y - 0.14, 0), min(y + 0.14, 1.0)]),
                marker=dict(colors=[self.tech_colors[key] for key in sorted_keys]),
                sort=False,
                title=dict(text=f"{title}",
                           position='middle center',
                           font=dict(size=20))
            )
            )

        feature_collection = FeatureCollection(feature_collection)

        fig.add_trace(go.Choropleth(
            locations=list(regions_dict.keys()),
            geojson=feature_collection,
            z=list(range(len(regions_dict.keys()))),
            text=list(regions_dict.keys()),
            marker=dict(opacity=0.3),
            colorscale='viridis',
            autocolorscale=False,
            reversescale=True,
            marker_line_color='black',
            marker_line_width=1.0,
        ))

        # Add offshore regions
        if 0:
            feature_collection = []
            for i, idx in enumerate(offshore_buses_index):

                region_shape = shapely.wkt.loads(self.net.buses.loc[idx].region)

                if isinstance(region_shape, sPolygon):
                    feature_collection += [Feature(geometry=Polygon([list(region_shape.exterior.coords)]), id=idx)]
                else:
                    feature_collection += \
                        [Feature(geometry=MultiPolygon([(list(poly.exterior.coords), ) for poly in region_shape]), id=idx)]

            feature_collection = FeatureCollection(feature_collection)

            fig.add_trace(go.Choropleth(
                locations=offshore_buses_index.values,
                geojson=feature_collection,
                z=list(range(len(offshore_buses_index))),
                text=offshore_buses_index,
                marker=dict(opacity=0.5),
                colorscale='plotly3',
                autocolorscale=False,
                reversescale=True,
                marker_line_color='black',
                marker_line_width=0.5,
            ))

        return fig

    def generate_pie_charts(self, region_name, regions):

        # Get buses in region
        buses_index = self.net.buses.loc[[idx for idx in self.net.buses.index if idx[2:4] in regions]].index

        # Get all generators for those buses
        generators = self.net.generators[self.net.generators.bus.isin(buses_index)]

        all_cap = dict.fromkeys(sorted(set(generators.type)))
        for key in all_cap:
            all_cap[key] = np.sum(self.net.generators_t.p[generators[generators.type == key].index].values)

        fig = go.Figure(
            data=go.Pie(
                values=list(all_cap.values()),
                labels=list(all_cap.keys()),
                hole=0.4,
                marker=dict(colors=[self.tech_colors[key] for key in all_cap.keys()]),
                title=dict(text=f"{region_name}<br>Generation (in GWh)",
                           position='middle center',
                           font=dict(size=20))))
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
    import yaml
    config = yaml.load(open(output_dir + "config.yaml", 'r'), yaml.SafeLoader)


    pprp = PyPSAResultsPlotly(output_dir)

    fig_choice = 6
    auto_open = True

    if fig_choice == 1:
        fig = pprp.show_topology()
        title = "topology.html"
    if fig_choice == 2:
        fig = pprp.show_topology_heatmap()
        title = "topology_heatmap.html"
    if fig_choice == 3:
        # Marginal prices
        case = "median"  # or "peak"
        fig = pprp.show_bus_marginal_price_choropleth(case)
        title = f"marginal_price_{case}.html"
    if fig_choice == 4:
        # Individual generator position
        types = ["wind_offshore"]  # ["wind_onshore", "wind_offshore", "pv_utility", "pv_residential"]
        attribute = "p_nom_opt"
        fig = pprp.show_generators(types, attribute)
        title = attribute + "_for_" + "_".join(types) + '.html'
    if fig_choice == 5:
        # Individual storage position
        types = ["sto", "phs"]
        attribute = "p_nom_opt"
        fig = pprp.show_storage(types, attribute)
        title = attribute + "_for_" + "_".join(types) + '.html'
    if fig_choice == 6:
        # Map with pie charts
        regions_dict = {'Iberia': ['ES', 'PT'],
                        'Central West': ['NL', 'BE', 'LU', 'FR', 'DE'],
                        'Nordics': ['DK', 'NO', 'SE', 'FI'],
                        'Baltics': ['LV', 'LT', 'EE'],
                        'British Isles': ['UK', 'IE'],
                        'Central South': ['CH', 'IT', 'AT', 'SI'],
                        'East': ['RO', 'PL', 'HU', 'CZ', 'SK'],
                        'South': ['HR', 'GR', 'AL', 'ME', 'BA', 'RS', 'BG']}
        fig = pprp.get_map_divided_by_region(regions_dict, config["res"]["strategy"])
        title = 'divide_map.html'

    fig.write_html(output_dir + title, auto_open=auto_open)


