import sys
import os
import pickle

import pandas as pd

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cf

from pyggrid.data.geographics import display_polygons
from pyggrid.resite.resite import Resite


def plot_map(resite: Resite, sites_data: pd.Series, data_name: str):
    """Plots a heat map of the given data for each technology."""

    show_shapes = True
    sites_shapes = resite.initial_sites_ds
    sites_index = sites_data.index
    for tech in set(sites_index.get_level_values(0)):
        tech_sites_index = sites_index[sites_index.get_level_values(0) == tech]
        points = list(zip(tech_sites_index.get_level_values(1), tech_sites_index.get_level_values(2)))
        xs, ys = zip(*points)
        if show_shapes:
            ax = display_polygons(sites_shapes[tech].values, fill=False, show=False)
        else:
            land_50m = cf.NaturalEarthFeature('physical', 'land', '50m',
                                              edgecolor='darkgrey',
                                              facecolor=cf.COLORS['land_alt1'])
            fig = plt.figure(figsize=(13, 13))
            ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
            ax.add_feature(land_50m, linewidth=0.5, zorder=-1)
            ax.add_feature(cf.BORDERS.with_scale('50m'), edgecolor='darkgrey', linewidth=0.5, zorder=0)

        ax.set_xlim(min(xs)-2, max(xs)+2)
        ax.set_ylim(min(ys)-2, max(ys)+2)
        sc = ax.scatter(xs, ys, marker='s', transform=ccrs.PlateCarree(), zorder=1, cmap='viridis',
                        c=sites_data[tech].values)
        plt.colorbar(sc)
        plt.title(f"{data_name} for {tech}")
        plt.show()


if __name__ == '__main__':

    assert (len(sys.argv) == 2) or (len(sys.argv) == 3), \
        "You need to provide one or two argument: output_dir (and test_number)"

    main_output_dir = sys.argv[1]
    test_number = sys.argv[2] if len(sys.argv) == 3 else None
    if test_number is None:
        test_number = sorted(os.listdir(main_output_dir))[-1]
    output_dir_ = f"{main_output_dir}{test_number}/"
    # output_dir_ = "/output/resite_EU_meet_res_agg_use_ex_cap/0.1/"
    print(output_dir_)

    resite_ = pickle.load(open(f"{output_dir_}resite_instance.p", 'rb'))
    print(f"Region: {resite_.regions}")

    data = resite_.sel_data_dict["cap_potential_ds"] * resite_.y_ds
    data_name_ = "Capacity Potential (GW)"
    plot_map(resite_, data, data_name_)
