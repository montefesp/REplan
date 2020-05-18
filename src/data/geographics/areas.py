from os.path import join, dirname, abspath

import pandas as pd


def get_nuts_area() -> pd.DataFrame:
    """Return for each NUTS region (2013 and 2016 version) its size in km2"""

    area_fn = join(dirname(abspath(__file__)), "../../../data/geographics/source/eurostat/reg_area3.xls")
    return pd.read_excel(area_fn, header=9, index_col=0)[:2193]
