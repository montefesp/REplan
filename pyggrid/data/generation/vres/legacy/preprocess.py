from os.path import join, dirname, abspath
from typing import List
import warnings

import pandas as pd

from pyggrid.data.geographics.grid_cells import get_grid_cells
from pyggrid.data.geographics import get_shapes, match_points_to_regions, convert_country_codes
from pyggrid.data.technologies import get_config_values


def get_legacy_capacity_in_regions_from_non_open(tech: str, regions_shapes: pd.Series, countries: List[str],
                                                 match_distance: float = 50., raise_error: bool = True) -> pd.Series:
    """
    Return the total existing capacity (in GW) for the given tech for a set of geographical regions.

    This function is using proprietary data.

    Parameters
    ----------
    tech: str
        Technology name.
    regions_shapes: pd.Series [Union[Polygon, MultiPolygon]]
        Geographical regions
    countries: List[str]
        List of ISO codes of countries in which the regions are situated
    match_distance: float (default: 50)
        Distance threshold (in km) used when associating points to shape.
    raise_error: bool (default: True)
        Whether to raise an error if no legacy data is available for this technology.

    Returns
    -------
    capacities: pd.Series
        Legacy capacities (in GW) of technology 'tech' for each region

    """

    path_legacy_data = join(dirname(abspath(__file__)), '../../../../../data/generation/vres/legacy/source/')

    capacities = pd.Series(0., index=regions_shapes.index)
    plant, plant_type = get_config_values(tech, ["plant", "type"])
    if (plant, plant_type) in [("Wind", "Onshore"), ("Wind", "Offshore"), ("PV", "Utility")]:

        if plant == "Wind":

            data = pd.read_excel(f"{path_legacy_data}Windfarms_Europe_20200127.xls", sheet_name='Windfarms',
                                 header=0, usecols=[2, 5, 9, 10, 18, 23], skiprows=[1], na_values='#ND')
            data = data.dropna(subset=['Latitude', 'Longitude', 'Total power'])
            data = data[data['Status'] != 'Dismantled']
            if countries is not None:
                data = data[data['ISO code'].isin(countries)]

            if len(data) == 0:
                return capacities

            # Converting from kW to GW
            data['Total power'] *= 1e-6
            data["Location"] = data[["Longitude", "Latitude"]].apply(lambda x: (x.Longitude, x.Latitude), axis=1)

            # Keep only onshore or offshore point depending on technology
            if plant_type == 'Onshore':
                data = data[data['Area'] != 'Offshore']
            else:  # Offshore
                data = data[data['Area'] == 'Offshore']

            if len(data) == 0:
                return capacities

        else:  # plant == "PV":

            data = pd.read_excel(f"{path_legacy_data}Solarfarms_Europe_20200208.xlsx", sheet_name='ProjReg_rpt',
                                 header=0, usecols=[0, 4, 8])
            data = data[pd.notnull(data['Coords'])]
            data["Location"] = data["Coords"].apply(lambda x: (float(x.split(',')[1]), float(x.split(',')[0])))
            if countries is not None:
                data['Country'] = convert_country_codes(data['Country'].values, 'name', 'alpha_2')
                data = data[data['Country'].isin(countries)]

            if len(data) == 0:
                return capacities

            # Converting from MW to GW
            data['Total power'] = data['MWac']*1e-3

        data = data[["Location", "Total power"]]

        points_region = match_points_to_regions(data["Location"].values, regions_shapes,
                                                distance_threshold=match_distance).dropna()

        for region in regions_shapes.index:
            points_in_region = points_region[points_region == region].index.values
            capacities[region] = data[data["Location"].isin(points_in_region)]["Total power"].sum()

    elif (plant, plant_type) == ("PV", "Residential"):

        legacy_capacity_fn = join(path_legacy_data, 'SolarEurope_Residential_deployment.xlsx')
        data = pd.read_excel(legacy_capacity_fn, header=0, index_col=0, usecols=[0, 4], squeeze=True).sort_index()
        data = data[data.index.isin(countries)]

        if len(data) == 0:
            return capacities

        # Get countries shapes
        countries_shapes = get_shapes(data.index.values, which='onshore', save=True)["geometry"]

        for region_id, region_shape in regions_shapes.items():
            for country_id, country_shape in countries_shapes.items():
                capacities[region_id] += \
                    (region_shape.intersection(country_shape).area/country_shape.area) * data[country_id]

    else:
        if raise_error:
            raise ValueError(f"Error: No legacy data exists for tech {tech} with plant {plant} and type {plant_type}.")
        else:
            warnings.warn(f"Warning: No legacy data exists for tech {tech}.")

    return capacities


def aggregate_legacy_capacity(spatial_resolution: float):
    """
    Aggregate legacy data at a given spatial resolution.

    Parameters
    ----------
    spatial_resolution: float
        Spatial resolution at which we want to aggregate.

    """

    countries = ["AL", "AT", "BA", "BE", "BG", "BY", "CH", "CY", "CZ", "DE", "DK", "EE", "ES",
                 "FI", "FO", "FR", "GB", "GR", "HR", "HU", "IE", "IS", "IT", "LT", "LU", "LV",
                 "ME", "MK", "NL", "NO", "PL", "PT", "RO", "RS", "SE", "SI", "SK", "UA"]

    technologies = ["wind_onshore", "wind_offshore", "pv_utility", "pv_residential"]

    capacities_df_ls = []
    for country in countries:
        print(f"Country: {country}")
        shapes = get_shapes([country])
        onshore_shape = shapes[~shapes["offshore"]]["geometry"].values[0]
        offshore_shape = shapes[shapes["offshore"]]["geometry"].values
        # If not offshore shape for country, remove offshore technologies from set
        offshore_shape = None if len(offshore_shape) == 0 else offshore_shape[0]
        technologies_in_country = technologies
        if offshore_shape is None:
            technologies_in_country = [tech for tech in technologies if get_config_values(tech, ['onshore'])]

        # Divide shapes into grid cells
        grid_cells_ds = get_grid_cells(technologies_in_country, spatial_resolution, onshore_shape, offshore_shape)
        technologies_in_country = set(grid_cells_ds.index.get_level_values(0))

        # Get capacity in each grid cell
        capacities_per_country_ds = pd.Series(index=grid_cells_ds.index, name="Capacity (GW)")
        for tech in technologies_in_country:
            capacities_per_country_ds[tech] = \
                get_legacy_capacity_in_regions_from_non_open(tech, grid_cells_ds.loc[tech].reset_index()[0], [country],
                                                             match_distance=100)
        capacities_per_country_df = capacities_per_country_ds.to_frame()
        capacities_per_country_df.loc[:, "ISO2"] = country
        capacities_df_ls += [capacities_per_country_df]

    # Aggregate dataframe from each country
    capacities_df = pd.concat(capacities_df_ls).sort_index()

    # Replace technology name by plant and type
    tech_to_plant_type = {tech: get_config_values(tech, ["plant", "type"]) for tech in technologies}
    capacities_df = capacities_df.reset_index()
    capacities_df["Plant"] = capacities_df["Technology Name"].apply(lambda x: tech_to_plant_type[x][0])
    capacities_df["Type"] = capacities_df["Technology Name"].apply(lambda x: tech_to_plant_type[x][1])
    capacities_df = capacities_df.drop("Technology Name", axis=1)
    capacities_df = capacities_df.set_index(["Plant", "Type", "Longitude", "Latitude"])

    legacy_dir = join(dirname(abspath(__file__)), '../../../../../data/generation/vres/legacy/generated/')
    capacities_df.round(4).to_csv(f"{legacy_dir}aggregated_capacity.csv",
                                  header=True, columns=["ISO2", "Capacity (GW)"])


if __name__ == '__main__':
    aggregate_legacy_capacity(0.25)
