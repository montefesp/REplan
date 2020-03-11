import os
import yaml
import pickle
import numpy as np

import dash
import dash_table
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
import plotly.graph_objs as go

from pypsa import Network

import sys
sys.path.append("../py_grid_exp/")


tech_colors = {"All": "rgba(138,43,226,0.5)",
               "nuclear": "rgba(255,140,0,0.5)",
               "wind_aerodyn": "rgba(30,144,255,0.5)",
               "solar_tallmaxm": "rgba(220,20,60,0.5)",
               "ccgt": "rgba(47,79,79,0.5)",
               "ocgt": "rgba(105,105,105,0.5)",
               "imports": "rgba(255,215,0,0.5)",
               "storage": "rgba(34,139,34,0.5)"}

# TODO: Maybe I should put each 'figure' into objects where I can just update some parts of
#  it so that the uploading is faster

class App:
    
    def __init__(self, output_dir, test_number=None):

        # Load css
        css_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets/")
        self.app = dash.Dash(__name__, assets_url_path=css_folder)

        # Load net
        # If no test_number is specified, take the last run
        self.current_test_number = test_number
        if self.current_test_number is None:
            self.current_test_number = sorted(os.listdir(output_dir))[-1]
        self.output_dir = output_dir
        self.net = Network()
        self.net.import_from_csv_folder(self.output_dir + self.current_test_number + "/")
        self.current_line_id = self.net.lines.index[0]
        self.current_bus_id = self.net.buses.index[0]

        self.selected_types = sorted(list(set(self.net.generators.type.values)))

    def built_app(self):
    
        def get_map():

            map_coords = [min(self.net.buses["x"].values) - 5,
                          max(self.net.buses["x"].values) + 5,
                          min(self.net.buses["y"].values) - 2,
                          max(self.net.buses["y"].values) + 2]
    
            fig = go.Figure(layout=go.Layout(
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
                            ))
    
            # Adding lines to map
            # Get minimum s_nom_opt
            s_nom_opt_min = min(self.net.lines.s_nom_opt[self.net.lines.s_nom_opt > 0].values)
            for i, idx in enumerate(self.net.lines.index):
                bus0_id = self.net.lines.loc[idx, ("bus0", )]
                bus1_id = self.net.lines.loc[idx, ("bus1", )]
                bus0_x = self.net.buses.loc[bus0_id, ("x", )]
                bus0_y = self.net.buses.loc[bus0_id, ("y", )]
                bus1_x = self.net.buses.loc[bus1_id, ("x", )]
                bus1_y = self.net.buses.loc[bus1_id, ("y", )]
                color = 'rgba(0,0,255,0.8)'
                name = 'AC'
                s_nom_mul = self.net.lines.loc[idx, ('s_nom_opt', )]/s_nom_opt_min
                if self.net.lines.loc[idx, ("carrier", )] == "DC":
                    color = 'rgba(255,0,0,0.8)'
                    name = 'DC'

                fig.add_trace(go.Scattergeo(
                        mode='lines',
                        lon=[bus0_x, (bus0_x+bus1_x)/2, bus1_x],
                        lat=[bus0_y, (bus0_y+bus1_y)/2, bus1_y],
                        line=dict(
                            width=np.log(1+s_nom_mul),
                            color=color),
                        text=[idx, idx, idx],
                        hoverinfo='text',
                        name=name
                    ))
    
            # Add points to map
            p_noms = np.zeros((len(self.net.buses.index, )))
            color = tech_colors['All']
            if len(self.selected_types) == 1:
                color = tech_colors[self.selected_types[0]]
            colors = [color]*len(self.net.buses.index)
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
                                text=self.net.buses.index,
                                hoverinfo='text',
                                marker=dict(
                                    size=10+40*np.log(1+p_noms/p_nom_max),
                                    color=colors
                                ),
                                name='bus'
                            ))

            return fig
    
        def get_line_info():
    
            fig = go.Figure(data=go.Scatter(
                                x=[i for i in range(len(self.net.snapshots))],
                                y=self.net.lines['s'].sel(id=self.current_line_id).values,
                                marker=dict(color='blue'),
                                name='Power flow'),
                            layout=go.Layout(
                                title='Power flow for ' + self.current_line_id,
                                xaxis={'title': 'Time stamps'},
                                yaxis={'title': 'GWh (or GW)'}))
    
            # Original capacity
            capacity = self.net.lines['s_nom'].sel(id=self.current_line_id).values.item()
            fig.add_trace(go.Scatter(
                x=[i for i in range(len(self.net.snapshots))],
                y=[capacity]*len(self.net.snapshots),
                marker=dict(color='red'),
                name='Original capacity'
            ))
    
            fig.add_trace(go.Scatter(
                x=[i for i in range(len(self.net.snapshots))],
                y=[-capacity]*len(self.net.snapshots),
                marker=dict(color='red', opacity=0.5),
                name='Original capacity'
            ))
    
            # New capacity
            capacity = self.net.lines['s_nom_opt'].sel(id=self.current_line_id).values.item()
            fig.add_trace(go.Scatter(
                x=[i for i in range(len(self.net.snapshots))],
                y=[capacity] * len(self.net.snapshots),
                marker=dict(color='green'),
                name='Updated capacity'
            ))
    
            fig.add_trace(go.Scatter(
                x=[i for i in range(len(self.net.snapshots))],
                y=[-capacity] * len(self.net.snapshots),
                marker=dict(color='green'),
                name='Updated capacity'
            ))
    
            return fig
    
        # Application layout
        def get_demand_balancing():
    
            # Get net import
            lines_out = self.net.lines.where(self.net.lines.bus0 == self.current_bus_id, drop=True)
            lines_in = self.net.lines.where(self.net.lines.bus1 == self.current_bus_id, drop=True)
            outflow = np.sum(lines_out.s.values, axis=0)
            inflow = np.sum(lines_in.s.values, axis=0)
            net_inflow = inflow - outflow
            pos_net_inflow = net_inflow.copy()
            pos_net_inflow[pos_net_inflow < 0] = 0
            neg_net_inflow = net_inflow.copy()
            neg_net_inflow[neg_net_inflow > 0] = 0

            # Get net output from battery
            storages = self.net.storages.where(self.net.storages.bus == self.current_bus_id, drop=True)
            storage_pos_discharge = -np.sum(storages.charge.values, axis=0)
            storage_pos_discharge[storage_pos_discharge < 0] = 0

            installed_cap = np.sum(self.net.generators['p_nom_opt']
                                   .where(self.net.generators.bus == self.current_bus_id, drop=True).values)
            available_power_per_gen = \
                np.array(self.net.generators['p_nom_opt']
                         .where(self.net.generators.bus == self.current_bus_id, drop=True).values) * \
                np.array(self.net.generators['p_max_pu']
                         .where(self.net.generators.bus == self.current_bus_id, drop=True).values).T
            available_power_per_bus = np.sum(available_power_per_gen.T, axis=0)
    
            # Compute total load, first line is useful in case there is no load at the selected bus
            load = np.zeros(len(self.net.snapshots))
            load += np.sum(self.net.loads['p_set']
                           .where(self.net.loads.bus == self.current_bus_id, drop=True).values, axis=0)
            fig = go.Figure(
                data=go.Scatter(
                    x=[i for i in range(len(self.net.snapshots))],
                    y=load,
                    name='Load'),
                layout=go.Layout(
                    title='Demand balancing in ' + self.current_bus_id,
                    xaxis={'title': 'Time stamps'},
                    yaxis={'title': 'GWh (or GW)',
                           'range': [0, 1.1 * np.max(
                               [np.max(load),
                                np.max(
                                    np.sum(self.net.generators['p']
                                           .where(self.net.generators.bus == self.current_bus_id, drop=True).values,
                                           axis=0)
                                    + pos_net_inflow),
                                installed_cap])]},
                ))
    
            fig.add_trace(go.Scatter(
                x=[i for i in range(len(self.net.snapshots))],
                y=[installed_cap]*len(self.net.snapshots),
                name='Gen Capacity'))
    
            fig.add_trace(go.Scatter(
                x=[i for i in range(len(self.net.snapshots))],
                y=available_power_per_bus,
                name='Available Cap'))
    
            generations = self.net.generators[['p', 'type']]\
                .where(self.net.generators.bus == self.current_bus_id, drop=True)
            types = list(set(generations.type.values))
            # Put nuclear first if present
            if 'nuclear' in types:
                types.remove("nuclear")
                types.insert(0, "nuclear")
            for t in types:
                total_generation = [0] * len(self.net.snapshots)
                generations_by_type = generations['p'].where(generations.type == t, drop=True).values
                if len(generations_by_type) != 0:
                    total_generation = np.sum(generations_by_type, axis=0)
                fig.add_trace(go.Scatter(
                        x=[i for i in range(len(self.net.snapshots))],
                        y=total_generation,
                        opacity=0.5,
                        stackgroup='one',
                        mode='none',
                        fillcolor=tech_colors[t],
                        marker=dict(color=tech_colors[t],
                                    opacity=0.5),
                        name=t))
    
            fig.add_trace(go.Scatter(
                    x=[i for i in range(len(self.net.snapshots))],
                    y=pos_net_inflow,
                    stackgroup='one',
                    mode='none',
                    marker=dict(opacity=0.5),
                    opacity=0.5,
                    fillcolor=tech_colors['imports'],
                    name='Pos Net Import'))

            fig.add_trace(go.Scatter(
                    x=[i for i in range(len(self.net.snapshots))],
                    y=storage_pos_discharge,
                    stackgroup='one',
                    mode='none',
                    marker=dict(opacity=0.5),
                    opacity=0.5,
                    fillcolor=tech_colors['storage'],
                    name='Storage Net Discharge'))

            return fig

        def get_state_of_charge():

            # Get storages at the current node
            storages = self.net.storages\
                .where(self.net.storages.bus == self.current_bus_id, drop=True)

            # State of charge
            fig = go.Figure(
                data=go.Scatter(
                    x=[i for i in range(len(self.net.snapshots))],
                    y=np.sum(storages.state_of_charge.values, axis=0),
                    name='SOF'),
                layout=go.Layout(
                    title='State of charge in ' + self.current_bus_id,
                    xaxis={'title': 'Time stamps'},
                    yaxis={'title': 'GWh (or GW)'},
                ))

            # Maximum level of charge
            fig.add_trace(go.Scatter(
                x=[i for i in range(len(self.net.snapshots))],
                y=[storages.p_nom_opt.values.item()*storages.max_hours.values.item()]*len(self.net.snapshots),
                name='Max Storage'
            ))

            return fig

        def get_charge_discharge():

            # Get storages at the current node
            storages = self.net.storages\
                .where(self.net.storages.bus == self.current_bus_id, drop=True)

            # State of charge
            fig = go.Figure(
                data=go.Scatter(
                    x=[i for i in range(len(self.net.snapshots))],
                    y=np.sum(storages.charge.values, axis=0),
                    name='Charge-Discharge'),
                layout=go.Layout(
                    title='Charge in ' + self.current_bus_id,
                    xaxis={'title': 'Time stamps'},
                    yaxis={'title': 'GWh (or GW)'},
                ))

            # Maximum level of charge
            fig.add_trace(go.Scatter(
                x=[i for i in range(len(self.net.snapshots))],
                y=[storages.p_nom_opt.values.item()]*len(self.net.snapshots),
                name='Max in power'
            ))

            # Maximum level of discharge
            fig.add_trace(go.Scatter(
                x=[i for i in range(len(self.net.snapshots))],
                y=[-storages.p_nom_opt.values.item()]*len(self.net.snapshots),
                name='Max out power'
            ))

            return fig

        def get_costs_table():

            #costs_fn = self.output_dir + self.current_test_number + "/costs.yaml"
            #costs = yaml.load(open(costs_fn, "r"), Loader=yaml.FullLoader)

            # Generation
            gen_types = set(self.net.generators.type.values)
            total_new_gen_cap = dict.fromkeys(gen_types, 0)
            gen_invest_cost = dict.fromkeys(gen_types, 0)
            gen_op_cost = dict.fromkeys(gen_types, 0)
            for idx in self.net.generators.index:
                gen = self.net.generators.loc[idx]
                t = gen.type
                invest_cost = gen.capital_cost
                new_gen_cap = gen.p_nom_opt - gen.p_nom
                total_new_gen_cap[t] += new_gen_cap
                gen_invest_cost[t] += new_gen_cap * invest_cost
                gen_op_cost[t] += np.sum(self.net.generators_t.p[idx]) * gen.marginal_cost

            # Transmission
            trans_invest_cost = 0
            total_new_trans_cap = 0
            for idx in self.net.lines.index:
                line = self.net.lines.loc[idx]
                new_trans_cap = line.s_nom_opt - line.s_nom
                total_new_trans_cap += new_trans_cap
                trans_invest_cost += new_trans_cap * line.capital_cost

            # Storage
            store_invest_cost = 0
            total_new_store_cap = 0
            for idx in self.net.storage_units.index:
                storage = self.net.storage_units.loc[idx]
                new_store_cap = storage.p_nom_opt - storage.p_nom
                total_new_store_cap += new_store_cap
                store_invest_cost += new_store_cap * storage.capital_cost

            # Lost load
            """
            lost_load_cost = 0
            lost_load_cost_per_unit = costs["lost_load"]
            for idx in self.net.buses.index:
                load = self.net.loads_t.where(self.net.loads.bus == idx).p_set.values[0]
                generation = np.sum(self.net.generators_t
                                    .where(self.net.generators.bus == idx).p.values, axis=0)
                inflows = np.sum(self.net.lines_t.where(self.net.lines.bus1 == idx).s.values, axis=0)
                outflows = np.sum(self.net.lines_t.where(self.net.lines.bus0 == idx).s.values, axis=0)
                battery_charge = np.sum(self.net.storages_t.where(self.net.storages.bus == idx).charge.values,
                                        axis=0)
                lost_load = np.sum(load - generation - inflows + outflows + battery_charge)
                print(lost_load)
                lost_load_cost += lost_load * lost_load_cost_per_unit
            """
            table = []

            diviser = 1000.0
            for t in gen_types:
                gen_invest_cost[t] /= diviser
                gen_op_cost[t] /= diviser
            total_gen_invest_cost = sum([gen_invest_cost[t] for t in gen_types])
            total_gen_op_cost = sum([gen_op_cost[t] for t in gen_types])
            total_gen_cost = total_gen_invest_cost + total_gen_op_cost
            trans_invest_cost /= diviser
            store_invest_cost /= diviser
            #lost_load_cost /= diviser

            for t in gen_types:
                table.append({"Tech": t, "Investment": "{0:.4f}".format(gen_invest_cost[t]),
                              "Operation": "{0:.4f}".format(gen_op_cost[t]),
                              "Total": "{0:.4f}".format(gen_invest_cost[t] + gen_op_cost[t])})
            table.append({"Tech": "Gen Total", "Investment": "{0:.4f}".format(total_gen_invest_cost),
                          "Operation": "{0:.4f}".format(total_gen_op_cost),
                          "Total": "{0:.4f}".format(total_gen_cost)})
            table.append({"Tech": "Trans Total", "Investment": "{0:.4f}".format(trans_invest_cost),
                          "Operation": 0,
                          "Total": "{0:.4f}".format(trans_invest_cost)})
            table.append({"Tech": "Store Total", "Investment": "{0:.4f}".format(store_invest_cost),
                          "Operation": 0,
                          "Total": "{0:.4f}".format(store_invest_cost)})
            #table.append({"Tech": "Lost load", "Investment": "N/A", "Operation": "N/A",
            #              "Total": "{0:.4f}".format(lost_load_cost)})
            table.append({"Tech": "Total", "Investment": "{0:.4f}".format(trans_invest_cost+total_gen_invest_cost),
                          "Operation": "{0:.4f}".format(total_gen_op_cost),
                          "Total": "{0:.4f}".format(trans_invest_cost+total_gen_cost+store_invest_cost)})

            return table

        def get_output_folders():
            return [{'label': dir_name, 'value': dir_name} for dir_name in sorted(os.listdir(self.output_dir))]
    
        def get_parameters_list():
            attrs = yaml.load(open(self.output_dir + self.current_test_number + "/tech_parameters.yaml", "rb"),
                              Loader=yaml.FullLoader)
            return [{'name': key, 'value': str(attrs[key])} for key in attrs]

        def get_tech_type_list():
            return [{'label': t, 'value': t} for t in self.selected_types]

        def get_capacities():

            all_cap = dict.fromkeys(sorted(set(self.net.generators.type)))
            for key in all_cap:
                all_cap[key] = np.sum(self.net.generators[self.net.generators.type == key].p_nom_opt.values)/1000.0

            fig = go.Figure(
                data=go.Pie(
                    title="Capacities (GW)",
                    values=list(all_cap.values()),
                    labels=list(all_cap.keys()),
                    marker=dict(colors=[tech_colors[key] for key in all_cap.keys()])))
            return fig

        def get_capacities_for_node():

            gens_at_bus = self.net.generators[self.net.generators.bus == self.current_bus_id]
            all_types = sorted(set(gens_at_bus.type.values))
            data = []
            for tech in all_types:
                data.append(go.Bar(name=tech,
                                   x=["Capacity"],
                                   y=[np.sum(gens_at_bus[self.net.generators.type == tech].p_nom_opt.values)/1000.0],
                                   marker=dict(color=tech_colors[tech])))

            fig = go.Figure(data=data)
            fig.update_layout(barmode='stack')

            return fig

        def get_generation():

            all_cap = dict.fromkeys(sorted(set(self.net.generators.type)))
            for key in all_cap:
                all_cap[key] = np.sum(self.net.generators_t.p[
                                          self.net.generators[self.net.generators.type == key].index].values)/1000.0

            fig = go.Figure(
                data=go.Pie(
                    title="Generation (GWh)",
                    values=list(all_cap.values()),
                    labels=list(all_cap.keys()),
                    marker=dict(colors=[tech_colors[key] for key in all_cap.keys()])))
            return fig

        def get_generation_for_node():

            gens_at_bus = self.net.generators[self.net.generators.bus == self.current_bus_id]
            all_types = sorted(set(gens_at_bus.type.values))
            data = []
            for tech in all_types:
                data.append(go.Bar(name=tech,
                                   x=["Generation"],
                                   y=[np.sum(self.net.generators_t.p[gens_at_bus[gens_at_bus.type == tech].index].values)/1000.0],
                                   marker=dict(color=tech_colors[tech])))

            fig = go.Figure(data=data)
            fig.update_layout(barmode='stack')

            return fig

        def get_load_gen():

            # Get the total load for every time stamp
            total_load = np.sum(self.net.loads_t.p_set.values, axis=1)

            # Get the total gen for every time stamp
            total_gen = np.zeros(len(total_load))
            techs = self.selected_types
            for tech in techs:
                total_gen += np.sum(self.net.generators_t.p[
                                        self.net.generators[self.net.generators.type == tech].index].values,
                                    axis=1)

            fig = go.Figure(
                data=go.Scatter(
                    x=[i for i in range(len(self.net.snapshots))],
                    y=total_load,
                    name='Total Load'),
                layout=go.Layout(
                    title='Total Load vs Total Gen',
                    xaxis={'title': 'Time stamps'},
                    yaxis={'title': 'MWh (or MW)'},
                ))

            # Maximum level of charge
            fig.add_trace(go.Scatter(
                x=[i for i in range(len(self.net.snapshots))],
                y=total_gen,
                name='Generation'
            ))

            return fig

        self.app.layout = html.Div([
    
            html.H1('Interactive network representation'),

            html.Div([

                # Parameters and costs
                html.Div([
                    dcc.Dropdown(
                        id='output-selector',
                        options=get_output_folders(),
                        value=self.current_test_number
                    ),

                    html.Div([
                        dash_table.DataTable(
                            id='tech_parameters-table',
                            columns=[{"name": "name", "id": "name"}, {"name": "value", "id": "value"}],
                            data=get_parameters_list()),

                        dcc.Checklist(
                            id="tech-types",
                            options=get_tech_type_list(),
                            value=self.selected_types
                        ),


                        html.Button(id='submit-button', n_clicks=0, children='Submit')

                    ]),

                    html.Div([
                        dash_table.DataTable(
                            id='cost-table',
                            columns=[{"name": "Tech", "id": "Tech"},
                                     {"name": "Investment (G€)", "id": "Investment"},
                                     {"name": "Operation (G€)", "id": "Operation"},
                                     {"name": "Total (G€)", "id": "Total"}],
                            data=get_costs_table()),

                    ])
                ], style={"width": 1000}),
                
                # Graphs
                html.Div([
                    # Map and general info
                    html.Div([

                        # Map
                        html.Div([
                            dcc.Graph(
                                id='map',
                                figure=get_map(),
                                style={'height': 700}
                            )
                        ]),

                        html.Div([
                            dcc.Graph(
                                id='tot-cap',
                                figure=get_capacities(),
                            ),

                            dcc.Graph(
                                id='tot-gen',
                                figure=get_generation(),
                            )
                        ]),

                        dcc.Graph(
                            id='load-gen',
                            figure=get_load_gen(),
                        )
                    ], style={'display': 'flex'}),

                    # Per-node info
                    html.Div([
                        """
                        html.Div([

                            # Line power flow
                            dcc.Graph(
                                id='line-power-flow',
                                figure=get_line_info(),
                            ),
                
                            # Demand balancing graph
                            dcc.Graph(
                                id='demand-balancing',
                                figure=get_demand_balancing(),
                            )

                        ]),

                        html.Div([
                            # Demand balancing graph
                            dcc.Graph(
                                id='state-of-charge',
                                figure=get_state_of_charge()
                            ),

                            # Demand balancing graph
                            dcc.Graph(
                                id='charge-discharge',
                                figure=get_charge_discharge()
                            )
                        ]),
                        """,

                        html.Div([
                            dcc.Graph(
                                id='cap-per-bus',
                                figure=get_capacities_for_node(),
                            ),

                            dcc.Graph(
                                id='gen-per-bus',
                                figure=get_generation_for_node(),
                            )
                        ])

                    ], style={'display': 'flex'})
                ], style={'display': 'flex', 'flex-wrap': 'wrap'})
            ])
        ])

        @self.app.callback(
            [# Output('demand-balancing', 'figure'),
             # Output('state-of-charge', 'figure'),
             # Output('charge-discharge', 'figure'),
             Output('cap-per-bus', 'figure'),
             Output('gen-per-bus', 'figure')],
            [Input('map', 'clickData')])
        def update_demand_balancing(clickData):
            if clickData is not None and 'points' in clickData:
                points = clickData['points'][0]
                if 'text' in points and '-' not in points['text']:
                    self.current_bus_id = points['text']
                    return get_capacities_for_node(), get_generation_for_node() #get_demand_balancing(), get_state_of_charge(), get_charge_discharge(), \
            raise PreventUpdate

        """
        @self.app.callback(
            Output('line-power-flow', 'figure'),
            [Input('map', 'clickData')])
        def update_line_power_flow(clickData):
            if clickData is not None and 'points' in clickData:
                points = clickData['points'][0]
                if 'text' in points and '-' in points['text']:
                    self.current_line_id = points['text']
                    return get_line_info()
            raise PreventUpdate
        """

        @self.app.callback(
            Output('tech_parameters-table', 'data'),
            [Input('output-selector', 'value')])
        def update_param_table(value):
            self.current_test_number = value
            return get_parameters_list()

        @self.app.callback(
            [Output('map', 'figure'),
             Output('cost-table', 'data'),
             Output('tot-cap', 'figure'),
             Output('tot-gen', 'figure'),
             Output('load-gen', 'figure')],
            [Input('submit-button', 'n_clicks')],
            [State('output-selector', 'value'),
             State('tech-types', 'value')])
        def update_network(n_clicks, value1, value2):
            print(value2)
            self.net = Network()
            self.net.import_from_csv_folder(self.output_dir + value1 + "/")
            self.current_line_id = self.net.lines.index[0]
            self.current_bus_id = self.net.buses.index[0]
            self.selected_types = value2
            return get_map(), get_costs_table(), get_capacities(), get_generation(), get_load_gen()

        return self.app


if __name__ == "__main__":

    assert (len(sys.argv) == 2) or (len(sys.argv) == 3), \
        "You need to provide one or two argument: output_dir (and test_number)"
    if len(sys.argv) == 2:
        app_object = App(sys.argv[1])
    else:
        app_object = App(sys.argv[1], sys.argv[2])
    app = app_object.built_app()
    app.run_server(debug=True)
