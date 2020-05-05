from os.path import join, dirname, abspath
import time

import pandas as pd
import numpy as np

import requests


def download_iea_electricity_production(countries, start_year, end_year):

    data_dir = join(dirname(abspath(__file__)), "../../../data/")
    countries_dict = pd.read_csv(f"{data_dir}countries-codes.csv", index_col="Code")

    years = list(range(start_year, end_year + 1))
    for country in countries:
        print(country)
        iea_country = countries_dict.loc[country]['IAE']
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
        production.to_csv(f"{data_dir}generation/source/iea/{country}.csv")


if __name__ == '__main__':

    countries_ = ["CY", "MT"]
    download_iea_electricity_production(countries_, 1990, 2017)
