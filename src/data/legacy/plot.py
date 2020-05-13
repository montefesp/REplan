from typing import List

import pandas as pd
import plotly.graph_objects as go

from src.data.geographics import convert_country_codes, get_subregions
from src.data.legacy import get_legacy_capacity_in_countries


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
    df["ISO_3"] = [convert_country_codes('alpha_3', alpha_2=c) for c in df["ISO"]]

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


if __name__ == '__main__':
    regions = get_subregions("EU2")
    plot_capacity_per_country("wind_offshore", regions, lonrange=[-12, 30], latrange=[35, 75])
