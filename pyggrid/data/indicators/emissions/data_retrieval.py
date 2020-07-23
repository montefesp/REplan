from typing import List
import time
import os

import pandas as pd

import requests


def download_iea_co2_by_electricity_and_heat(countries: List[str]):

    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../data/")
    countries_df = pd.read_csv(f"{data_dir}geographics/countries-codes.csv", index_col="Code")

    for country in countries:
        production = pd.DataFrame(columns=["CO2 from electricity and heat producers (MT)"])
        iea_country = countries_df.loc[country]['IAE']
        url = f"https://api.iea.org/stats/indicator/CO2BySector?countries={iea_country}"
        print(url)
        time.sleep(1)

        # Get the html content of the page
        table = requests.get(url).json()
        for line in table:
            if line['flowLabel'] == 'Electricity and heat producers':
                production.loc[int(line['year']), "CO2 from electricity and heat producers (MT)"] = line['value']

        print(production)

        production.to_csv(f"{data_dir}emission/source/iea/{country}.csv")


if __name__ == '__main__':

    countries_ = ["CY", "MT"]
    download_iea_co2_by_electricity_and_heat(countries_)
