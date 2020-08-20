from typing import List

import pandas as pd
import plotly.graph_objects as go

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cf

from pyggrid.data.geographics import convert_country_codes, get_subregions
from pyggrid.data.generation.vres.legacy import get_legacy_capacity_in_countries
from pyggrid.data.technologies import get_config_values

def plot_capacity_per_country(tech: str, countries: List[str],
                              lonrange=None, latrange=None) -> None:
    """
    Plot a choropleth map of the existing capacity of a technology in a series of countries.

    Parameters
    ----------
    tech: str
        One of wind_offshore, wind_onshore, pv_utility, pv_residential
    countries: List[str]
        List of ISO codes

    """

    ds = get_legacy_capacity_in_countries(tech, countries)
    df = pd.DataFrame({"ISO": ds.index, "Capacity": ds.values})
    df["ISO_3"] = convert_country_codes(df["ISO"].values, 'alpha_2', 'alpha_3')

    fig = go.Figure(data=go.Choropleth(
        locations=df['ISO_3'],  # Spatial coordinates
        z=df['Capacity'],  # Data to be color-coded
        text=[f"{cap} GW" for cap in df["Capacity"].values],
        colorscale='Reds',
        colorbar_title="Capacity (GW)"
    ))

    fig.update_layout(
        geo=dict(
            lonaxis=dict(
                range=lonrange,
            ),
            lataxis=dict(
                range=latrange,
            ),
            scope='europe')
    )

    fig.show()


def plot_per_point(tech: str, show: bool = True):

    # TODO: revise
    plant, plant_type = get_config_values(tech, ["plant", "type"])
    capacities_df = pd.read_csv("/home/utilisateur/Global_Grid/code/pyggrid/data/generation/vres/legacy/"
                                "generated/aggregated_capacity_harmonized.csv",
                                index_col=[0, 1]).loc[plant].loc[plant_type]
    capacities_df = capacities_df[capacities_df["ISO2"] != 'IS']
    capacities_df = capacities_df[capacities_df["Capacity (GW)"] != 0.0]

    land_50m = cf.NaturalEarthFeature('physical', 'land', '50m',
                                      edgecolor='darkgrey',
                                      facecolor=cf.COLORS['land_alt1'])

    fig = plt.figure(figsize=(13, 13))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    ax.add_feature(land_50m, linewidth=0.5, zorder=-1)
    ax.add_feature(cf.BORDERS.with_scale('50m'), edgecolor='darkgrey', linewidth=0.5, zorder=-1)
    ax.set_extent([-15, 42.5, 30, 72.5])

    map = ax.scatter(capacities_df["Longitude"], capacities_df["Latitude"], c=capacities_df["Capacity (GW)"],
               s=1, vmax=1.2, vmin=0.0)
    fig.colorbar(map)

    if show:
        plt.show()
    else:
        return ax


def plot_diff(tech: str, show: bool = True):

    plant, plant_type = get_config_values(tech, ["plant", "type"])
    capacities_df = pd.read_csv("/home/utilisateur/Global_Grid/code/pyggrid/data/generation/vres/legacy/"
                                "generated/aggregated_capacity.csv",
                                index_col=[0, 1]).loc[plant].loc[plant_type]
    capacities_df = capacities_df[capacities_df["ISO2"] != 'IS']
    capacities_df = capacities_df[capacities_df["Capacity (GW)"] != 0.0]

    capacities_df_h = pd.read_csv("/home/utilisateur/Global_Grid/code/pyggrid/data/generation/vres/legacy/"
                                  "generated/aggregated_capacity_harmonized.csv",
                                  index_col=[0, 1]).loc[plant].loc[plant_type]
    capacities_df_h = capacities_df_h[capacities_df_h["ISO2"] != 'IS']
    capacities_df_h = capacities_df_h[capacities_df_h["Capacity (GW)"] != 0.0]

    capacities_df["Difference (GW)"] = capacities_df["Capacity (GW)"] - capacities_df_h["Capacity (GW)"]

    land_50m = cf.NaturalEarthFeature('physical', 'land', '50m',
                                      edgecolor='darkgrey',
                                      facecolor=cf.COLORS['land_alt1'])

    fig = plt.figure(figsize=(13, 13))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    ax.add_feature(land_50m, linewidth=0.5, zorder=-1)
    ax.add_feature(cf.BORDERS.with_scale('50m'), edgecolor='darkgrey', linewidth=0.5, zorder=-1)
    ax.set_extent([-15, 42.5, 30, 72.5])

    map = ax.scatter(capacities_df["Longitude"], capacities_df["Latitude"], c=capacities_df["Difference (GW)"],
               s=1, vmax=1.2, vmin=0.0)
    fig.colorbar(map)

    if show:
        plt.show()
    else:
        return ax

if __name__ == '__main__':
    regions = get_subregions("EU2")
    #plot_capacity_per_country("wind_offshore", regions, lonrange=[-12, 30], latrange=[35, 75])
    # plot_per_point("wind_onshore")
    plot_diff("wind_onshore")