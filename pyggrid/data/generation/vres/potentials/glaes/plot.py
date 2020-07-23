from typing import List

from pyggrid.data.generation.vres.potentials.glaes import get_capacity_potential_per_country
from pyggrid.data.geographics import convert_country_codes
from pyggrid.data.technologies import get_config_dict

import plotly.graph_objects as go


def plot_capacity(tech: str, countries: List[str],
                  lon_range: List[float] = None, lat_range: List[float] = None):
    """
    Plot on a map potential capacity (in GW) per country for a specific technology.

    Parameters
    ----------
    tech: str
        Technology name.
    countries: List[str]
        List of ISO codes of countries.
    lon_range: List[float] (default: None)
        Longitudinal range over which to display the map. Automatically set if not specified.
    lat_range: List[float] (default: None)
        Latitudinal range over which to display the map. Automatically set if not specified.

    """

    tech_config_dict = get_config_dict([tech], ["onshore", "power_density", "filters"])[tech]
    cap_pot_ds = get_capacity_potential_per_country(countries, tech_config_dict["onshore"],
                                                    tech_config_dict["filters"], tech_config_dict["power_density"])
    cap_pot_df = cap_pot_ds.to_frame()
    cap_pot_df["ISO_3"] = convert_country_codes(cap_pot_df.index.values, 'alpha_2', 'alpha_3')

    fig = go.Figure(data=go.Choropleth(
        locations=cap_pot_df['ISO_3'],  # Spatial coordinates
        z=cap_pot_df[0],  # Data to be color-coded
        text=[f"{cap} GW" for cap in cap_pot_df[0].values],
        colorscale='Reds',
        colorbar_title=f"Capacity (GW)"
    ))

    fig.update_layout(
        geo=dict(
            lonaxis=dict(
                range=lon_range,
            ),
            lataxis=dict(
                range=lat_range,
            ),
            scope='europe'),
        title=f"Capacity potential for {tech}"
    )

    fig.show()


if __name__ == '__main__':
    from pyggrid.data.geographics import get_subregions
    countries_ = get_subregions("BENELUX")
    tech_ = "pv_residential"
    plot_capacity(tech_, countries_, lon_range=[-12, 30], lat_range=[35, 75])
