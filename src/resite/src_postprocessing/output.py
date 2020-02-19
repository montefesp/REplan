import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt
import pandas as pd
from os.path import join, abspath
from src.tools import read_database, get_global_coordinates, return_output, return_coordinates_from_countries, \
                            selected_data, retrieve_load_data
from src_postprocessing.output_tools import read_output, read_inputs_plotting, plot_basemap, \
                                                assess_firmness, clip_revenue, \
                                                assess_capacity_credit, return_coordinates
from itertools import chain, combinations, cycle, islice
from random import randint
from copy import deepcopy
from numpy import array, tile, ceil, sum, arange, fromiter
from xarray import concat
import matplotlib.dates as mdates


class Output(object):

    def __init__(self, path):

        self.output_folder_path = abspath(join('../output_data/', path))
        self.parameters_yaml = read_inputs_plotting(self.output_folder_path)
        self.outputs_pickle = read_output(path)

    def return_numerics(self, choice, ndiff=1, firm_threshold=0.3):
        """Function returning statistical measures associatd with the aggregation
        of different resource time series.

            Parameters:

            ------------

            choice : list
                The time series aggregation to assess.
                    "opt" - the (optimal) selection associated with the suggested siting problem
                    "max" - the selection of n locations maximizing production
                    "rand" - the selection of n random sites

            ndiff : int
                Defines the nth difference to be computed within time series.

            firm_threshold : float
                Defines the threshold that defines the "firmness" of time series

            Returns:

            ------------

            result_dict : dict
                Dictionary containing various indexed indicators to be plotted.
        """

        print('Problem: {}'.format(self.parameters_yaml['main_problem'], end=', '))
        print('Objective: {}'.format(self.parameters_yaml['main_objective'], end=', '))
        print('Spatial resolution: {}'.format(self.parameters_yaml['spatial_resolution'], end=', '))
        print('Time horizon: {}'.format(self.parameters_yaml['time_slice'], end=', '))
        print('Measure: {}'.format(self.parameters_yaml['resource_quality_measure'], end=', '))
        print('alpha: {}'.format(self.parameters_yaml['alpha_rule'], end=', '))
        print('delta: {}'.format(self.parameters_yaml['delta'], end=', '))
        print('beta: {}'.format(self.parameters_yaml['beta']))

        path_resource_data = self.parameters_yaml['path_resource_data'] + str(self.parameters_yaml['spatial_resolution']) + '/'
        path_transfer_function_data = self.parameters_yaml['path_transfer_function_data']
        path_load_data = self.parameters_yaml['path_load_data']

        horizon = self.parameters_yaml['time_slice']
        delta = self.parameters_yaml['delta']
        technologies = self.parameters_yaml['technologies']
        regions = self.parameters_yaml['regions']
        no_sites = self.parameters_yaml['cardinality_constraint']

        price_ts = pd.read_csv('../input_data/el_price/elix_2014_2018.csv', index_col=0, sep=';')
        price_ts = price_ts['price']
        length_timeseries = self.outputs_pickle['capacity_factors_dict'][technologies[0]].shape[0]

        if length_timeseries <= price_ts.size:
            price_ts = price_ts[:length_timeseries]
        else:
            el_ts_multiplier = int(ceil(length_timeseries / price_ts.size))
            price_ts = tile(price_ts, el_ts_multiplier)
            price_ts = price_ts[:length_timeseries]

        # TODO: too many parameters
        load_ts = retrieve_load_data(path_load_data, horizon, delta, regions,
                                     alpha_plan='centralized', alpha_load_norm='min')

        signal_dict = dict.fromkeys(choice, None)
        firmness_dict = dict.fromkeys(choice, None)
        difference_dict = dict.fromkeys(choice, None)
        result_dict = dict.fromkeys(['signal', 'difference', 'firmness'], None)

        database = read_database(path_resource_data)
        global_coordinates = get_global_coordinates(database, self.parameters_yaml['spatial_resolution'],
                                                    self.parameters_yaml['population_density_threshold'],
                                                    self.parameters_yaml['protected_areas_selection'],
                                                    self.parameters_yaml['protected_areas_threshold'],
                                                    self.parameters_yaml['altitude_threshold'],
                                                    self.parameters_yaml['slope_threshold'],
                                                    self.parameters_yaml['forestry_threshold'],
                                                    self.parameters_yaml['depth_threshold'],
                                                    self.parameters_yaml['path_population_density_data'],
                                                    self.parameters_yaml['path_protected_areas_data'],
                                                    self.parameters_yaml['path_orography_data'],
                                                    self.parameters_yaml['path_landseamask'],
                                                    population_density_layer=self.parameters_yaml['population_density_layer'],
                                                    protected_areas_layer=self.parameters_yaml['protected_areas_layer'],
                                                    orography_layer=self.parameters_yaml['orography_layer'],
                                                    forestry_layer=self.parameters_yaml['forestry_layer'],
                                                    bathymetry_layer=self.parameters_yaml['bathymetry_layer'])

        for c in choice:

            if c == 'COMP':

                array_list = []
                for tech in self.outputs_pickle['optimal_location_dict'].keys():
                    array_per_tech = array(self.outputs_pickle['capacity_factors_dict'][tech].sel(
                        locations=self.outputs_pickle['optimal_location_dict'][tech]).values).sum(axis=1)
                    array_list.append(array_per_tech)

            elif c == 'RAND':

                region_coordinates = return_coordinates_from_countries(regions, global_coordinates, add_offshore=True)
                truncated_data = selected_data(database, region_coordinates, horizon)
                output_data = return_output(truncated_data, technologies, path_transfer_function_data)

                no_coordinates = sum(fromiter((len(lst) for lst in region_coordinates.values()), dtype=int))

                output = []
                for item in output_data.keys():
                    output.append(output_data[item])

                output_overall = concat(output, dim='locations')

                score_init = 0.
                for i in range(10000):

                    location_list = []
                    ts_list = []

                    idx = [randint(0, no_coordinates*len(technologies) - 1) for x in range(sum(no_sites))]
                    for loc in idx:
                        location_list.append(output_overall.isel(locations=loc).locations.values.flatten()[0])
                        ts_list.append(output_overall.isel(locations=loc).values)

                    score = array(ts_list).sum()

                    if score > score_init:
                        score_init = score
                        ts_incumbent = ts_list

                array_list = ts_incumbent

            elif c == 'PROD':

                suboptimal_dict = dict.fromkeys(self.parameters_yaml['regions'], None)
                suboptimal_dict_ts = deepcopy(suboptimal_dict)

                if len(no_sites) == 1:

                    location_list = []
                    ts_list = []
                    truncated_data_list = []
                    output_data_list = []

                    for key in suboptimal_dict.keys():

                        region_coordinates = return_coordinates_from_countries(key, global_coordinates,
                                                                               add_offshore=self.parameters_yaml[
                                                                                   'add_offshore'])
                        truncated_data = selected_data(database, region_coordinates, horizon)

                        for k in technologies:

                            tech = []
                            tech.append(k)

                            output_data = return_output(truncated_data, tech, path_transfer_function_data)[k]
                            output_data_list.append(output_data)

                            truncated_data_sum = output_data.sum(dim='time')
                            truncated_data_list.append(truncated_data_sum)

                    truncated_data_concat = concat(truncated_data_list, dim='locations')
                    output_data_concat = concat(output_data_list, dim='locations')

                    tdata = truncated_data_concat.argsort()[-no_sites[0]:]

                    for loc in tdata.values:
                        location_list.append(output_data_concat.isel(locations=loc).locations.values.flatten()[0])
                        ts_list.append(output_data_concat.isel(locations=loc).values)

                    array_list = ts_list

                else:

                    idx = 0

                    for key in suboptimal_dict.keys():

                        location_list = []
                        ts_list = []
                        output_data_list = []
                        truncated_data_list_per_region = []

                        region_coordinates = return_coordinates_from_countries(key, global_coordinates,
                                                                               add_offshore=self.parameters_yaml[
                                                                                   'add_offshore'])
                        truncated_data = selected_data(database, region_coordinates, horizon)

                        for k in technologies:

                            tech = []
                            tech.append(k)

                            output_data = return_output(truncated_data, tech, path_transfer_function_data)[k]
                            output_data_list.append(output_data)

                            truncated_data_sum = output_data.sum(dim='time')
                            truncated_data_list_per_region.append(truncated_data_sum)

                        truncated_data_concat_per_region = concat(truncated_data_list_per_region, dim='locations')
                        output_data_concat = concat(output_data_list, dim='locations')

                        tdata = truncated_data_concat_per_region.argsort()[-no_sites[idx]:]

                        for loc in tdata.values:
                            location_list.append(output_data_concat.isel(locations=loc).locations.values.flatten()[0])
                            ts_list.append(output_data_concat.isel(locations=loc).values)

                        idx += 1

                        suboptimal_dict[key] = location_list
                        suboptimal_dict_ts[key] = ts_list

                    array_list = []
                    for region in suboptimal_dict_ts.keys():
                        array_per_tech = array(suboptimal_dict_ts[region]).sum(axis=0)
                        array_list.append(array_per_tech)

            elif c == 'NSEA':

                location_list = []
                ts_list = []

                region_coordinates = return_coordinates('NSea', global_coordinates)
                truncated_data = selected_data(database, region_coordinates, horizon)
                output_data = return_output(truncated_data, technologies, path_transfer_function_data)

                truncated_data_sum = output_data['wind_aerodyn'].sum(dim='time')

                tdata = truncated_data_sum.argsort()[-no_sites[0]:]

                for loc in tdata.values:
                    location_list.append(output_data['wind_aerodyn'].isel(locations=loc).locations.values.flatten()[0])
                    ts_list.append(output_data['wind_aerodyn'].isel(locations=loc).values)

                array_list = ts_list

            array_sum = pd.Series(data=array(array_list).sum(axis=0))
            difference = array_sum.diff(periods=ndiff).dropna()
            firmness = assess_firmness(array_sum, firm_threshold * sum(self.parameters_yaml['cardinality_constraint']))

            print('-------------------------------------')
            print('NUMERICAL RESULTS FOR THE {} SET OF SITES.'.format(str(c)))

            print('Variance of time series: {}'.format(round(array_sum.var(), 4)))
            print('Mean of time series: {}'.format(round(array_sum.mean(), 4)))
            print('Mean +/- std of time series: {}'.format(round(array_sum.mean() - array_sum.std(), 4)))
            print('p1, p5, p10 of time series: {}, {}, {}'.format(round(array_sum.quantile(q=0.01), 4),
                                                                  round(array_sum.quantile(q=0.05), 4),
                                                                  round(array_sum.quantile(q=0.1), 4)))
            print('{} difference count within +/- 1%, 5% of total output: {}, {}'.format(ndiff, difference.between(
                left=-0.01 * sum(self.parameters_yaml['cardinality_constraint']),
                right=0.01 * sum(self.parameters_yaml['cardinality_constraint'])).sum(), difference.between(
                left=-0.05 * sum(self.parameters_yaml['cardinality_constraint']),
                right=0.05 * sum(self.parameters_yaml['cardinality_constraint'])).sum()))
            print('Total estimated revenue: {}'.format(round(price_ts * array_sum).sum(), 4))
            print('Estimated low-yield (10, 20, 30%) revenue : {}, {}, {}'.format(
                round(clip_revenue(array_sum, price_ts, 0.1), 4),
                round(clip_revenue(array_sum, price_ts, 0.2), 4),
                round(clip_revenue(array_sum, price_ts, 0.3), 4)))
            print('Estimated capacity credit for top (1, 5, 10%) peak demand hours '
                  '(valid for centralized planning only): {}, {}, {}'.format(
                    round(assess_capacity_credit(load_ts, array_sum,
                                                 sum(self.parameters_yaml['cardinality_constraint']), 0.99), 4),
                    round(assess_capacity_credit(load_ts, array_sum,
                                                 sum(self.parameters_yaml['cardinality_constraint']), 0.95), 4),
                    round(assess_capacity_credit(load_ts, array_sum,
                                                 sum(self.parameters_yaml['cardinality_constraint']), 0.90), 4))
            )

            signal_dict[c] = array_sum
            difference_dict[c] = difference
            firmness_dict[c] = firmness

        result_dict['signal'] = signal_dict
        result_dict['difference'] = difference_dict
        result_dict['firmness'] = firmness_dict
        result_dict['no_sites'] = sum(no_sites)

        return result_dict

    def plot_numerics(self, result_dict, plot_cdf=False, plot_boxplot=True, plot_signal=False, plot_cdfirm=True,
                      signal_range=[0, 8760]):

        """Plotting different results.

            Parameters:

            ------------

            result_dict : dict
                Dictionary containing various indexed indicators to be plotted.

            plot_cdf : boolean
                Plot the cdf in its own frame.

            boxplot : boolean
                Plot boxplot in its own frame.


            plot_signal : boolean
                Plot the resource time series in its own frame.

            signal_range : list
                List of integers defining range of signal to be displayed.

            one_plot : boolean
                Plot all subplots in one frame.

        """
        if plot_cdf:

            colors = islice(cycle(('royalblue', 'crimson', 'forestgreen', 'goldenrod')), 0,
                            len(result_dict['signal'].keys()))
            plt.clf()
            ax = plt.subplot(111)

            for c in list({str(j) for i in result_dict.values() for j in i}):
                ax.hist(result_dict['signal'][c], bins=1000, density=True, cumulative=True, label=str(c),
                        color=next(colors), histtype='step', alpha=0.8, linewidth=1.0)

            ax.set_xlabel('Aggregated output [-]')
            ax.set_xticks(array([0., 10., 15., 20.]))
            ax.set_ylabel('Probability [-]')
            ax.set_yticks(arange(0.2, 1.1, step=0.2))

            ax.legend(fontsize='medium', frameon=False, bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower left",
                      mode="expand", borderaxespad=0, ncol=len(result_dict['signal'].keys()))

            plt.savefig('cdf.pdf', dpi=200, bbox_inches='tight')
            plt.show()

        if plot_signal:
            colors = islice(cycle(('royalblue', 'crimson', 'forestgreen', 'goldenrod')), 0,
                            len(result_dict['signal'].keys()))
            plt.clf()
            for c in result_dict['signal'].keys():

                plt.plot(range(signal_range[0], signal_range[1]),
                         result_dict['signal'][c][signal_range[0]:signal_range[1]], label=str(c), color=next(colors))
            plt.legend(loc='best')
            plt.show()

        if plot_cdfirm:

            colors = islice(cycle(('royalblue', 'crimson', 'forestgreen', 'goldenrod')), 0,
                            len(result_dict['signal'].keys()))
            linestyles = islice(cycle(('-', '--', '-.', ':')), 0, len(result_dict['signal'].keys()))

            plt.clf()

            fig = plt.figure(figsize=(7, 4))

            ax1 = fig.add_subplot(121)
            ax1.set_xlabel('Aggregated output [-]', fontsize=10)
            ax1.set_ylabel('Probability [-]', fontsize=10)
            ax1.set_yticks(arange(0.2, 1.1, step=0.2))

            ax2 = fig.add_subplot(122)
            ax2.set_xlabel('Firm window length [h]', fontsize=10)
            ax2.set_ylabel('Occurrences [-]', fontsize=10)
            ax2.set_yscale('log', nonposy='clip')

            for c in result_dict['signal'].keys():

                color = next(colors)
                linestyle = next(linestyles)

                ax1.hist(result_dict['signal'][c], bins=1000, density=True, cumulative=True, label=str(c), color=color,
                         histtype='step', alpha=1.0, linewidth=1.0, linestyle=linestyle)
                ax2.hist(result_dict['firmness'][c], bins=50, label=str(c), color=color, cumulative=False, alpha=1.0,
                         histtype='step', linewidth=1.0, linestyle=linestyle)

            # ax1.legend(fontsize='medium', frameon=False, bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower left",
            #            mode="expand", borderaxespad=0, ncol=len(result_dict['signal'].keys()))

            ax2.legend(fontsize='medium', frameon=False, loc="upper right",
                       mode="expand", borderaxespad=0, ncol=1)

            fig.tight_layout()
            fig.savefig(abspath(join(self.output_folder_path, 'numerics_plot.png')), bbox_inches='tight', dpi=300)

        if plot_boxplot:

            df = pd.DataFrame(columns=list(result_dict['signal'].keys()))

            for c in df.columns:
                df[str(c)] = result_dict['signal'][c]

            datetime_idx = pd.date_range('2008-01-01', periods=df.shape[0], freq='H')
            df.index = datetime_idx
            hour = df.index.hour

            df_filtered = df.iloc[(hour == 6) | (hour == 9) | (hour == 12) | (hour == 15) | (hour == 18) | (hour == 21)]

            fig, axs = plt.subplots(1, len(df.columns), figsize=(13,3), facecolor='w', edgecolor='k')
            fig.subplots_adjust(wspace=.0001)

            axs = axs.ravel()

            for i, name in enumerate(df.columns):
                axs[i].title.set_text(str(name))
                axs[i].set_xlabel('Hour of the day')
                axs[i].tick_params(axis='y', which='both', length=0)
                axs[i].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
                axs[i].set_ylim([-1, float(result_dict['no_sites'])+1])

                if i == 0:
                    axs[i].set_ylabel('Aggregated output [-]')

                if (i != 0) & (i != len(df.columns) - 1):
                    axs[i].set_yticklabels([])
                if i == len(df.columns) - 1:
                    axs[i].yaxis.tick_right()
                    axs[i].yaxis.set_label_position("right")
                    axs[i].set_ylabel('Aggregated output [-]')
                df_filtered.set_index(df_filtered.index.hour, append=True)[name].unstack().plot.box(ax=axs[i], sym='+',
                                                                                                    color=dict(
                                                                                                        boxes='royalblue',
                                                                                                        whiskers='royalblue',
                                                                                                        medians='forestgreen',
                                                                                                        caps='royalblue'),
                                                                                                    boxprops=dict(
                                                                                                        linestyle='-',
                                                                                                        linewidth=1.5),
                                                                                                    flierprops=dict(
                                                                                                        linewidth=1.5,
                                                                                                        markeredgecolor='crimson',
                                                                                                        color='crimson'),
                                                                                                    medianprops=dict(
                                                                                                        linestyle='-',
                                                                                                        linewidth=1.5),
                                                                                                    whiskerprops=dict(
                                                                                                        linestyle='-',
                                                                                                        linewidth=1.5),
                                                                                                    capprops=dict(
                                                                                                        linestyle='-',
                                                                                                        linewidth=1.5)
                                                                                                    )
                axs[i].yaxis.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.5)
            fig.tight_layout()
            fig.savefig(abspath(join(self.output_folder_path,
                                     'boxplot.pdf')), bbox_inches='tight', dpi=300)

    def optimal_locations_plot(self):

        """Plotting the optimal locations."""

        plt.clf()

        base = plot_basemap(self.outputs_pickle['coordinates_dict'])
        map = base['basemap']

        map.scatter(base['lons'], base['lats'], transform=base['projection'], marker='o', color='darkgrey', s=base['width']/1e7, zorder=2, alpha=1.0)

        tech_list = list(self.outputs_pickle['optimal_location_dict'].keys())
        tech_set = list(chain.from_iterable(combinations(tech_list, n) for n in range(1, len(tech_list)+1)))
        locations_plot = dict.fromkeys(tech_set, None)

        for key in locations_plot.keys():
            set_list = []

            for k in key:
                set_list.append(set(self.outputs_pickle['optimal_location_dict'][k]))
            locations_plot[key] = set.intersection(*set_list)

        for key in locations_plot.keys():
            proxy = set()
            init = locations_plot[key]
            subkeys = [x for x in tech_set if (x != key and len(x) > len(key))]

            if len(subkeys) > 0:

                for k in subkeys:
                    if proxy == set():
                        proxy = locations_plot[key].difference(locations_plot[k])
                    else:
                        proxy = proxy.difference(locations_plot[k])
                locations_plot[key] = list(proxy)

            else:
                locations_plot[key] = list(init)

        markers = islice(cycle(('.')), 0, len(locations_plot.keys()))
        colors = islice(cycle(('royalblue','crimson','forestgreen','goldenrod')), 0, len(locations_plot.keys()))
        for key in locations_plot.keys():

            longitudes = [i[0] for i in locations_plot[key]]
            latitudes = [i[1] for i in locations_plot[key]]
            map.scatter(longitudes, latitudes, transform=base['projection'], marker=next(markers), color=next(colors),
                        s=base['width']/(1e5), zorder=3, alpha=0.9, label=str(key))

        L = plt.legend(loc='upper left', fontsize='medium')
        L.get_texts()[0].set_text('Wind')
        L.get_texts()[1].set_text('PV')
        L.get_texts()[2].set_text('Wind & PV')

        plt.savefig(abspath(join(self.output_folder_path,
                                 'optimal_deployment.png')),
                                 bbox_inches='tight', dpi=300)

    def retrieve_max_locations(self):

        path_resource_data = self.parameters_yaml['path_resource_data'] + str(self.parameters_yaml['spatial_resolution']) + '/'
        path_transfer_function_data = self.parameters_yaml['path_transfer_function_data']

        horizon = self.parameters_yaml['time_slice']
        technologies = self.parameters_yaml['technologies']
        no_sites = self.parameters_yaml['cardinality_constraint']

        database = read_database(path_resource_data)
        global_coordinates = get_global_coordinates(database, self.parameters_yaml['spatial_resolution'],
                                                    self.parameters_yaml['population_density_threshold'],
                                                    self.parameters_yaml['protected_areas_selection'],
                                                    self.parameters_yaml['protected_areas_threshold'],
                                                    self.parameters_yaml['altitude_threshold'],
                                                    self.parameters_yaml['slope_threshold'],
                                                    self.parameters_yaml['forestry_threshold'],
                                                    self.parameters_yaml['depth_threshold'],
                                                    self.parameters_yaml['path_population_density_data'],
                                                    self.parameters_yaml['path_protected_areas_data'],
                                                    self.parameters_yaml['path_orography_data'],
                                                    self.parameters_yaml['path_landseamask'],
                                                    population_density_layer=self.parameters_yaml[
                                                        'population_density_layer'],
                                                    protected_areas_layer=self.parameters_yaml['protected_areas_layer'],
                                                    orography_layer=self.parameters_yaml['orography_layer'],
                                                    forestry_layer=self.parameters_yaml['forestry_layer'],
                                                    bathymetry_layer=self.parameters_yaml['bathymetry_layer'])

        suboptimal_dict = dict.fromkeys(self.parameters_yaml['regions'], None)

        location_list = []
        truncated_data_list = []
        output_data_list = []

        if len(no_sites) == 1:

            for key in suboptimal_dict.keys():

                region_coordinates = return_coordinates_from_countries(key, global_coordinates, add_offshore=False)
                truncated_data = selected_data(database, region_coordinates, horizon)

                for k in technologies:
                    tech = []
                    tech.append(k)

                    output_data = return_output(truncated_data, tech, path_transfer_function_data)[k]
                    truncated_data_sum = output_data.sum(dim='time')

                    truncated_data_list.append(truncated_data_sum)
                    output_data_list.append(output_data)

            truncated_data_concat = concat(truncated_data_list, dim='locations')
            output_data_concat = concat(output_data_list, dim='locations')

            tdata = truncated_data_concat.argsort()[-no_sites[0]:]

            for loc in tdata.values:
                location_list.append(output_data_concat.isel(locations=loc).locations.values.flatten()[0])

        else:

            raise ValueError(' Method not ready yet for partitioned problem.')

        return location_list

    def max_locations_plot(self, max_locations):

        """Plotting the optimal vs max. locations."""

        plt.clf()

        base_max = plot_basemap(self.outputs_pickle['coordinates_dict'])
        map_max = base_max['basemap']

        map_max.scatter(base_max['lons'], base_max['lats'], transform=base_max['projection'], marker='o',
                        color='darkgrey', s=base_max['width']/1e7, zorder=2, alpha=1.0)

        longitudes = [i[0] for i in max_locations]
        latitudes = [i[1] for i in max_locations]
        map_max.scatter(longitudes, latitudes, transform=base_max['projection'], marker='.', color='royalblue',
                        s=base_max['width']/(1e5), zorder=3, alpha=0.9, label='Wind')

        plt.savefig(abspath(join(self.output_folder_path,
                                 'suboptimal_deployment_'+str('&'.join(tuple(self.outputs_pickle['coordinates_dict'].keys())))+'.png')),
                                 bbox_inches='tight', dpi=300)

    def optimal_locations_plot_heatmaps(self):

        """Plotting the optimal location heatmaps."""

        if (self.parameters_yaml['problem'] == 'Covering' and self.parameters_yaml['objective'] == 'budget') or \
                (self.parameters_yaml['problem'] == 'Load' and self.parameters_yaml['objective'] == 'following'):

            for it in self.outputs_pickle['deployed_capacities_dict'].keys():

                plt.clf()

                base = plot_basemap(self.outputs_pickle['coordinates_dict'])
                map = base['basemap']

                location_and_capacity_list = []
                print([val for vals in self.outputs_pickle['coordinates_dict'].values() for val in vals])
                # for idx, location in enumerate(list(set([val for vals in self.outputs_pickle['coordinates_dict'].values() for val in vals]))):
                for idx, location in enumerate([val for vals in self.outputs_pickle['coordinates_dict'].values() for val in vals]):
                    l = list(location)
                    l.append(self.outputs_pickle['deployed_capacities_dict'][it][idx])
                    location_and_capacity_list.append(l)
                print(location_and_capacity_list)
                df = pd.DataFrame(location_and_capacity_list, columns=['lon', 'lat', 'cap'])

                pl = map.scatter(df['lon'].values, df['lat'].values, transform=base['projection'], c=df['cap'].values, marker='s', s=base['width']/1e6, cmap=plt.cm.Reds, zorder=2)

                cbar = plt.colorbar(pl, ax=map, orientation= 'horizontal', pad=0.1, fraction=0.04, aspect=28)
                cbar.set_label("GW", fontsize='8')
                cbar.outline.set_visible(False)
                cbar.ax.tick_params(labelsize='x-small')

                plt.savefig(abspath(join(self.output_folder_path, 'capacity_heatmap_'+str(it)+'.pdf')), bbox_inches='tight', dpi=300)

        else:
            raise TypeError('WARNING! No such plotting capabilities for a basic deployment problem.')
