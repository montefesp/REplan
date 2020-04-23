import matplotlib.pyplot as plt
from datetime import datetime
import pandas as pd
import yaml

from pypsa import Network

class SizingResultsSingleNet:

    """
    This class allows to extract results from a optimized PyPSA network.
    """

    def __init__(self, network: Network):

        self.net = network


    def make_plots_one_net(self):

        color_dict = {'wind_onshore': 'dodgerblue',
                      'wind_offshore': 'royalblue',
                      'wind_floating': 'navy',
                      'pv_residential': 'gold',
                      'pv_utility': 'goldenrod',
                      'load': 'darkred',
                      'ccgt': 'peru',
                      'ror': 'forestgreen',
                      'sto': 'darkgreen',
                      'demand': 'black',
                      'res': 'blue'}

        resolution = 'H'
        start = datetime(2015, 1, 1, 0, 0, 0)
        end = datetime(2015, 1, 31, 23, 0, 0)

        generation_t = self.net.generators_t['p']
        generation_techs = self.net.generators['type'].unique()

        print('test')

        self.plot_selected_generation(start, end, resolution, generation_t, generation_techs, color_dict)

    def plot_selected_generation(self, start, end, resolution,
                                 timeseries, techs, colors):

        demand_t = self.net.loads_t.p.resample(resolution).sum()
        demand_total = demand_t.sum(axis=1)
        demand_t_slice = demand_total[start:end]

        ccgt_t = timeseries.loc[:, timeseries.columns.str.contains('ccgt')].resample(resolution).sum()
        ccgt_t_total = ccgt_t.sum(axis=1)
        ccgt_t_slice = ccgt_t_total[start:end]

        ens_t = timeseries.loc[:, timeseries.columns.str.contains('Load shed')].resample(resolution).sum()
        ens_t_total = ens_t.sum(axis=1)
        ens_t_slice = ens_t_total[start:end]

        generation_total = pd.Series(0., index=demand_t_slice.index)

        fig, ax1 = plt.subplots()

        for item in techs:

            if item in ['wind_onshore', 'wind_offshore', 'wind_floating', 'pv_residential', 'pv_utility']:

                generation_t = timeseries.loc[:, timeseries.columns.str.contains(item)].resample(resolution).sum()
                generation_t_per_tech = generation_t.sum(axis=1)
                generation_t_slice = generation_t_per_tech[start:end]

                generation_total += generation_t_slice

        ax1.plot(generation_total.index, generation_total.values, color=colors['res'], label='demand', alpha=0.5)
        ax1.plot(demand_t_slice.index, demand_t_slice.values, color=colors['demand'], label='total res', alpha=0.5)
        ax1.set_ylabel('Demand and RES [GW]')

        ax2 = ax1.twinx()
        ax2.plot(ccgt_t_slice.index, ccgt_t_slice.values, color=colors['ccgt'], label='ccgt', alpha=0.9)
        ax2.plot(ens_t_slice.index, ens_t_slice.values, color=colors['load'], label='ens', alpha=0.9)
        ax2.set_ylabel('CCGT and ENS [GW]')

        fig.autofmt_xdate()

        ax1.legend(loc='upper left')
        ax2.legend(loc='upper right')

        plt.show()





class SizingResultsCompare:

    """
    This class allows to extract results from a optimized PyPSA network.
    """

    def __init__(self, first_network: Network, second_network: Network):

        self.first_net = first_network
        self.second_net = second_network


    def make_plots_compare(self):

        color_dict = {'wind_onshore': 'dodgerblue',
                      'wind_offshore': 'royalblue',
                      'wind_floating': 'navy',
                      'pv_residential': 'gold',
                      'pv_utility': 'goldenrod',
                      'load': 'darkred',
                      'ccgt': 'peru',
                      'ror': 'forestgreen',
                      'sto': 'darkgreen',
                      'demand': 'black',
                      'res': 'blue'}

        resolution = 'H'
        start = datetime(2017, 1, 1, 0, 0, 0)
        end = datetime(2017, 1, 30, 23, 0, 0)

        self.plot_res_profiles(start, end, resolution, color_dict)

    def plot_res_profiles(self, start, end, resolution, colors):

        techs = self.first_net.generators['type'].unique()

        first_generation_t = self.first_net.generators_t['p_max_pu']
        second_generation_t = self.second_net.generators_t['p_max_pu']

        fig, ax1 = plt.subplots()

        for item in ['wind']:

            if item in ['wind']:

                generation_t = first_generation_t.loc[:, first_generation_t.columns.str.contains(item)].resample(resolution).sum()
                generation_t_per_tech = generation_t.sum(axis=1)
                generation_t_slice = generation_t_per_tech[start:end]

                ax1.plot(generation_t_slice.index, generation_t_slice.values,
                         color='red', label=item+'_'+str(first_strategy), alpha=0.5, linestyle='--')

                generation_t = second_generation_t.loc[:, second_generation_t.columns.str.contains(item)].resample(resolution).sum()
                generation_t_per_tech = generation_t.sum(axis=1)
                generation_t_slice = generation_t_per_tech[start:end]

                ax1.plot(generation_t_slice.index, generation_t_slice.values,
                         color='blue', label=item+'_'+str(second_strategy), alpha=0.5, linestyle='-')

        ax1.set_ylabel('RES in-feed [sum of p.u.]')

        fig.autofmt_xdate()

        ax1.legend(loc='upper right')
        plt.show()



if __name__ == "__main__":

    topology = 'tyndp2018'
    main_output_dir = f'../../output/sizing/{topology}/'

    results = 'compare' # 'compare', 'single'

    if results == 'single':

        run_id = '20200418_224238'

        output_dir = f"{main_output_dir}{run_id}/"

        net = Network()
        net.import_from_csv_folder(output_dir)

        pprp = SizingResultsSingleNet(net)

        pprp.make_plots_one_net()

    else:

        first_run_id = '20200419_130741'
        second_run_id = '20200419_143433'

        first_output_dir = f"{main_output_dir}{first_run_id}/"
        second_output_dir = f"{main_output_dir}{second_run_id}/"

        first_config_file = yaml.load(open(f"{first_output_dir}config.yaml", "r"), Loader=yaml.FullLoader)
        first_strategy = [key for key, value in first_config_file['res']['strategies'].items() if (len(value) > 0) &
                          (key in ['comp', 'max'])][0]
        second_config_file = yaml.load(open(f"{second_output_dir}config.yaml", "r"), Loader=yaml.FullLoader)
        second_strategy = [key for key, value in second_config_file['res']['strategies'].items() if (len(value) > 0) &
                           (key in ['comp', 'max'])][0]

        first_net = Network()
        first_net.import_from_csv_folder(first_output_dir)

        second_net = Network()
        second_net.import_from_csv_folder(second_output_dir)

        pprp = SizingResultsCompare(first_net, second_net)

        pprp.make_plots_compare()