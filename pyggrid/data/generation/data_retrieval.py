from os.path import join, dirname, abspath
import time

import pandas as pd
import numpy as np

import requests


def download_iea_electricity_production(countries, start_year, end_year):

    data_dir = join(dirname(abspath(__file__)), "../../../data/")
    countries_df = pd.read_csv(f"{data_dir}geographics/countries-codes.csv", index_col="Code")

    years = list(range(start_year, end_year + 1))
    for country in countries:
        print(country)
        iea_country = countries_df.loc[country]['IAE']
        production = pd.DataFrame(columns=["Electricity Production (GWh)"], index=years)
        for year in range(start_year, end_year + 1):
            url = f"https://api.iea.org/stats/?year={year}&countries={iea_country}&series=ELECTRICITYANDHEAT"
            print(url)
            time.sleep(1)

            # Get the html content of the page
            table = requests.get(url).json()
            for line in table:
                if line["flowLabel"] == "Total production" and line["productLabel"] == "Electricity":
                    value = line['value']
                    if value == 0:
                        value = np.nan
                    production.loc[year, "Electricity Production (GWh)"] = value
                    break
        production.to_csv(f"{data_dir}generation/misc/source/iea/total/{country}.csv")


def download_iea_hydro_electricity_production(countries, start_year, end_year):

    data_dir = join(dirname(abspath(__file__)), "../../../data/")
    countries_df = pd.read_csv(f"{data_dir}geographics/countries-codes.csv", index_col="Code")

    years = list(range(start_year, end_year + 1))
    for country in countries:
        print(country)
        iea_country = countries_df.loc[country]['IAE']
        name = "Hydro Electricity Production (GWh)"
        production = pd.DataFrame(columns=[name], index=years)
        for year in range(start_year, end_year + 1):
            url = f"https://api.iea.org/stats/?year={year}&countries={iea_country}&series=ELECTRICITYANDHEAT"
            print(url)
            time.sleep(1)

            # Get the html content of the page
            table = requests.get(url).json()
            for line in table:
                if line["flowLabel"] == "Hydro" and line["productLabel"] == "Electricity":
                    value = line['value']
                    if value == 0:
                        value = np.nan
                    production.loc[year, name] = value
                    break
        production.to_csv(f"{data_dir}generation/misc/source/iea/hydro/{country}.csv")


if __name__ == '__main__':

    countries_ = ["BE"]
    download_iea_hydro_electricity_production(countries_, 2008, 2018)
