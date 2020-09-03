from os.path import join

import numpy as np
import pandas as pd


def return_filtered_and_normed(signal, delta, type='min'):
    # TODO: comment
    l_smooth = signal.rolling(window=delta, center=True).mean().dropna()
    if type == 'min':
        l_norm = (l_smooth - l_smooth.min()) / (l_smooth.max() - l_smooth.min())
    else:
        l_norm = l_smooth / l_smooth.max()

    return l_norm


# TODO: to be removed?
def retrieve_load_data_partitions(path_load_data, date_slice, alpha, delta, regions, norm_type):
    """Allows to get load for a whole region or per subregion"""
    dict_regions = {'EU': ['AT', 'BE', 'CH', 'DE', 'DK', 'ES',
                           'FR', 'UK', 'IE', 'IT', 'LU',
                           'NL', 'PT', 'SE', 'CZ',
                           'BG', 'CH', 'EE',
                           'FI', 'EL', 'HR', 'HU', 'LT', 'LV', 'PL', 'RO', 'SI', 'SK'],
                    'CWE': ['FR', 'BE', 'LU', 'NL', 'DE'],
                    'BL': ['BE', 'LU', 'NL']}  # TODO: BL is actually also the code of country

    load_data = pd.read_csv(join(path_load_data, 'load_opsd_2015_2018.csv'), index_col=0)
    # load_data.index = pd.date_range('2015-01-01T00:00', '2018-12-31T23:00', freq='H')
    load_data.index = pd.to_datetime(load_data.index)

    print(load_data)
    exit()

    load_data_sliced = load_data.loc[date_slice[0]:date_slice[1]]

    # Adding the stand-alone regions to load dict.
    standalone_regions = list(load_data.columns)
    for region in standalone_regions:
        dict_regions.update({str(region): str(region)})

    if alpha == 'load_central':

        # Extract lists of load subdivisions from load_dict.
        # e.g.: for regions ['BL', 'DE'] => ['BE', 'NL', 'LU', 'DE']
        regions_list = []
        for key in regions:
            if isinstance(dict_regions[key], str):
                regions_list.append(str(dict_regions[key]))
            elif isinstance(dict_regions[key], list):
                regions_list.extend(dict_regions[key])
            else:
                raise TypeError('Check again the type. Should be str or list.')

        load_vector = load_data_sliced[regions_list].sum(axis=1)

    elif alpha == 'load_partition':

        if regions in standalone_regions:
            load_vector = load_data_sliced[dict_regions[regions]]
        else:
            load_vector = load_data_sliced[dict_regions[regions]].sum(axis=1)

    load_vector_norm = return_filtered_and_normed(load_vector, delta, norm_type)

    return load_vector_norm


def resource_quality_mapping(cap_factor_df, delta, measure):
    # TODO: comment

    cap_factor_rolling = cap_factor_df.rolling(delta, center=True)

    if measure == 'mean':
        cap_factor_per_window_df = cap_factor_rolling.mean().dropna()
    elif measure == 'median':
        cap_factor_per_window_df = cap_factor_rolling.median().dropna()

    else:
        raise ValueError(' Measure {} is not available.'.format(str(measure)))

    return cap_factor_per_window_df


def critical_window_mapping(cap_factor_per_window_df, alpha, delta, regions, load_df, norm_type):
    # TODO: comment

    if alpha == 'load_central':

        load_df_sum = load_df.sum(axis=1)
        alpha_reference = return_filtered_and_normed(load_df_sum, delta, norm_type).to_frame()
        critical_windows = cap_factor_per_window_df.gt(alpha_reference.values).astype(int)

        return critical_windows

    elif alpha == 'load_partition':

        # TODO: update or remove
        for region, tech in key_list:
            l_norm = retrieve_load_data_partitions(path_load_data, date_slice, alpha, delta, region, norm_type)
            # Flip axes.
            alpha_reference = l_norm[:, np.newaxis]

            # Select region of interest within the dict value with 'tech' key.
            critical_windows = (input_dict[region][tech] > alpha_reference).astype(int)
            output_dict[region][tech] = critical_windows

    else:
        raise ValueError('No such alpha rule. Retry.')

    return output_dict
