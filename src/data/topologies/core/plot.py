import pandas as pd
import random

from shapely.geometry import Polygon, MultiPolygon

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature


def plot_topology(buses: pd.DataFrame, lines: pd.DataFrame = None) -> None:
    """
    Plot a map with buses and lines.

    Parameters
    ----------
    buses: pd.DataFrame
        DataFrame with columns 'x', 'y' and 'region'
    lines: pd.DataFrame (default: None)
        DataFrame with columns 'bus0', 'bus1' whose values must be index of 'buses'.
        If None, do not display the lines.
    """

    # Fill the countries with one color
    def get_xy(shape):
        # Get a vector of latitude and longitude
        xs = [i for i, _ in shape.exterior.coords]
        ys = [j for _, j in shape.exterior.coords]
        return xs, ys

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS, linestyle=':')

    # Plotting the buses
    for idx in buses.index:

        color = (random.random(), random.random(), random.random())

        # If buses are associated to regions, display the region
        if 'region' in buses.columns:
            region = buses.loc[idx].region
            if isinstance(region, MultiPolygon):
                for polygon in region:
                    x, y = get_xy(polygon)
                    ax.fill(x, y, c=color, alpha=0.3)
            elif isinstance(region, Polygon):
                x, y = get_xy(region)
                ax.fill(x, y, c=color, alpha=0.3)

        # Plot the bus position
        ax.scatter(buses.loc[idx].x, buses.loc[idx].y, c=[color], marker="s")

    # Plotting the lines
    if lines is not None:
        for idx in lines.index:

            bus0 = lines.loc[idx].bus0
            bus1 = lines.loc[idx].bus1
            if bus0 not in buses.index or bus1 not in buses.index:
                print(f"Warning: not showing line {idx} because missing bus {bus0} or {bus1}")
                continue

            color = 'r' if 'carrier' in lines.columns and lines.loc[idx].carrier == "DC" else 'b'
            plt.plot([buses.loc[bus0].x, buses.loc[bus1].x], [buses.loc[bus0].y, buses.loc[bus1].y], c=color)