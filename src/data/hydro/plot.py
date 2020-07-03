from typing import List

import pandas as pd
import plotly.graph_objects as go

from src.data.geographics import convert_country_codes, get_subregions
from src.data.hydro import get_hydro_capacities


def plot_capacity_per_country(tech: str, countries: List[str],
                              lon_range: List[float] = None, lat_range: List[float] = None) -> None:
    """
    Plot a choropleth map of the existing capacity of a technology in a series of countries.

    Parameters
    ----------
    tech: str
        One of ror, sto and phs
    countries: List[str]
        List of ISO codes
    lon_range: List[float] (default: None)
        Longitudinal range over which to display the map. Automatically set if not specified.
    lat_range: List[float] (default: None)
        Latitudinal range over which to display the map. Automatically set if not specified.
    """

    series = get_hydro_capacities("countries", tech)
    if not isinstance(series, tuple):
        series = (series, )
    for ds in series:
        df = pd.DataFrame(index=countries, columns=["ISO", "Capacity"])
        df.loc[ds.index, "Capacity"] = ds.values
        df["ISO_3"] = convert_country_codes(df.index.values, 'alpha_2', 'alpha_3')

        unit = ds.name.split(" ")[1]
        fig = go.Figure(data=go.Choropleth(
            locations=df['ISO_3'],  # Spatial coordinates
            z=df['Capacity'],  # Data to be color-coded
            text=[f"{cap} GW" for cap in df["Capacity"].values],
            colorscale='Reds',
            colorbar_title=f"Capacity {unit}"
        ))

        fig.update_layout(
            geo=dict(
                lonaxis=dict(
                    range=lon_range,
                ),
                lataxis=dict(
                    range=lat_range,
                ),
                scope='europe')
        )

        fig.show()


if __name__ == '__main__':
    regions = get_subregions("EU2")
    plot_capacity_per_country("ror", regions, lon_range=[-12, 30], lat_range=[35, 75])
