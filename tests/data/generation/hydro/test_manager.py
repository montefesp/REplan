import pytest

from pyggrid.data.generation.hydro.manager import *
from datetime import datetime


def get_timestamps(years: List[int]):
    start = datetime(years[0], 1, 1, 0, 0, 0)
    end = datetime(years[1], 12, 31, 23, 0, 0)
    return pd.date_range(start, end, freq='H')


def test_get_hydro_capacities_wrong_level():
    with pytest.raises(AssertionError):
        get_hydro_capacities("NUTS", "ror")


def test_get_hydro_capacities_wrong_plant_type():
    with pytest.raises(AssertionError):
        get_hydro_capacities("NUTS2", "nuclear")


def test_get_hydro_capacities():
    nb_per_level = {'ror': [26, 123],
                    'phs': [22, 77],
                    'sto': [29, 108]}
    for i, level in enumerate(["countries", "NUTS2"]):
        ds = get_hydro_capacities(level, "ror")
        assert isinstance(ds, pd.Series)
        assert ds.name == "ROR_CAP [GW]"
        assert len(ds) == nb_per_level['ror'][i]
        ds_cap, ds_en = get_hydro_capacities(level, "phs")
        assert isinstance(ds_cap, pd.Series)
        assert isinstance(ds_en, pd.Series)
        assert ds_cap.name == "PSP_CAP [GW]"
        assert len(ds_cap) == nb_per_level['phs'][i]
        assert ds_en.name == "PSP_EN_CAP [GWh]"
        assert len(ds_en) == nb_per_level['phs'][i]
        ds_cap, ds_en = get_hydro_capacities(level, "sto")
        assert isinstance(ds_cap, pd.Series)
        assert isinstance(ds_en, pd.Series)
        assert ds_cap.name == "STO_CAP [GW]"
        assert len(ds_cap) == nb_per_level['sto'][i]
        assert ds_en.name == "STO_EN_CAP [GWh]"
        assert len(ds_en) == nb_per_level['sto'][i]


def test_get_hydro_inflows_wrong_plant_type():
    with pytest.raises(AssertionError):
        get_hydro_inflows("NUTS2", "phs")


def test_get_hydro_inflows_wrong_level():
    with pytest.raises(AssertionError):
        get_hydro_inflows("NUTS", "ror")


def test_get_hydro_inflows_missing_timestamps():
    with pytest.raises(AssertionError):
        get_hydro_inflows("countries", "ror", get_timestamps([2030, 2030]))


def test_get_hydro_inflows():
    for i, level in enumerate(["countries", "NUTS2"]):
        df = get_hydro_inflows(level, "ror")
        ds = get_hydro_capacities(level, "ror")
        assert isinstance(df, pd.DataFrame)
        assert set(df.columns) == set(ds.index)
        df = get_hydro_inflows(level, "sto")
        ds_cap, ds_en = get_hydro_capacities(level, "sto")
        assert isinstance(df, pd.DataFrame)
        assert set(df.columns) == set(ds_cap.index)
        assert set(df.columns) == set(ds_en.index)


def test_phs_inputs_nuts_to_ehighway():
    ds_nuts_cap, ds_nuts_en = get_hydro_capacities("NUTS3", "phs")
    buses = get_ehighway_clusters().index
    ds_cap, ds_en = phs_inputs_nuts_to_ehighway(buses, ds_nuts_cap, ds_nuts_en)
    buses_without_phs = ['04ES', '07ES', '10ES', '14FR', '16FR', '17FR', '18FR', '19FR', '22FR', '23FR', '24FR',
                         '25FR', '26FR', '29LU', '30NL', '32DE', '38DK', '41PL', '51AT', '53IT', '55IT', '58HU',
                         '59RO', '60RO', '61RO', '63BA', '64ME', '67MK', '69GR', '70AL', '71UA', '72DK', '73EE',
                         '74FI', '75FI', '78LV', '80NO', '82NO', '84NO', '85NO', '86SE', '87SE', '89SE', '90UK',
                         '91UK', '93UK', '95UK', 'xxMA', 'xxTN', 'xxTR']
    assert set(buses) - set(ds_cap.index) == set(buses_without_phs)
    assert set(buses) - set(ds_en.index) == set(buses_without_phs)
    assert round(ds_cap.sum(), 3) == round(ds_nuts_cap.sum(), 3)
    assert round(ds_en.sum(), 3) == round(ds_nuts_en.sum(), 3)


def test_ror_inputs_nuts_to_ehighway():
    ds_nuts_cap = get_hydro_capacities("NUTS3", "ror")
    df_nuts_inf = get_hydro_inflows("NUTS3", "ror")
    buses = get_ehighway_clusters().index
    ds_cap, df_inf = ror_inputs_nuts_to_ehighway(buses, ds_nuts_cap, df_nuts_inf)
    buses_without_ror = ['01ES', '02ES', '03ES', '04ES', '07ES', '08ES', '09ES', '10ES', '11ES', '17FR', '18FR',
                         '21FR', '22FR', '23FR', '24FR', '26FR', '27FR', '29LU', '30NL', '32DE', '38DK', '40CZ',
                         '41PL', '42PL', '63BA', '71UA', '72DK', '73EE', '77LT', '79NO', '80NO', '81NO', '82NO',
                         '83NO', '84NO', '85NO', '89SE', '90UK', '91UK', '95UK', 'xxMA', 'xxTN', 'xxTR']
    assert set(buses) - set(ds_cap.index) == set(buses_without_ror)
    assert set(buses) - set(df_inf.columns) == set(buses_without_ror)
    assert round(ds_cap.sum(), 3) == round(ds_nuts_cap.sum(), 3)


def test_sto_inputs_nuts_to_ehighway():
    ds_nuts_cap, ds_nuts_en = get_hydro_capacities("NUTS3", "sto")
    df_nuts_inf = get_hydro_inflows("NUTS3", "sto")
    # Remove Corsica NUTS zones because they are not in eHighway
    ds_nuts_cap = ds_nuts_cap.drop(['FRM01', 'FRM02'])
    ds_nuts_en = ds_nuts_en.drop(['FRM01', 'FRM02'])
    df_nuts_inf = df_nuts_inf.drop(['FRM01', 'FRM02'], axis=1)

    buses = get_ehighway_clusters().index
    ds_cap, ds_en, df_inf = sto_inputs_nuts_to_ehighway(buses, ds_nuts_cap, ds_nuts_en, df_nuts_inf)
    buses_without_ror = ['17FR', '21FR', '22FR', '23FR', '24FR', '25FR', '26FR', '27FR', '29LU', '30NL', '31DE',
                         '32DE', '33DE', '34DE', '36DE', '37DE', '38DK', '51AT', '58HU', '63BA', '71UA', '72DK',
                         '73EE', '78LV', '89SE', '90UK', '91UK', '92UK', '93UK', '95UK', 'xxMA', 'xxTN', 'xxTR']
    assert set(buses) - set(ds_cap.index) == set(buses_without_ror)
    assert set(buses) - set(ds_en.index) == set(buses_without_ror)
    assert set(buses) - set(df_inf.columns) == set(buses_without_ror)
    assert round(ds_cap.sum(), 3) == round(ds_nuts_cap.sum(), 3)
    assert round(ds_en.sum(), 3) == round(ds_nuts_en.sum(), 3)
    assert round(df_inf.sum().sum(), 3) == round(df_nuts_inf.sum().sum(), 3)


def test_get_hydro_production_empty_years():
    with pytest.raises(AssertionError):
        get_hydro_production(years=[])


def test_get_hydro_production_empty_countries():
    with pytest.raises(AssertionError):
        get_hydro_production(countries=[])


def test_get_hydro_production_missing_year():
    with pytest.raises(AssertionError):
        get_hydro_production(years=[2000, 2050])


def test_get_hydro_production_missing_country():
    with pytest.raises(AssertionError):
        get_hydro_production(countries=["ZZ"])


def test_get_hydro_production_subset():
    countries = ["BE", "NL", "PL"]
    years = [2012, 2015, 2017]
    df = get_hydro_production(countries=countries, years=years)
    assert isinstance(df, pd.DataFrame)
    assert not (set(countries).symmetric_difference(set(df.index)))
    assert not (set(years).symmetric_difference(set(df.columns)))


def test_get_hydro_production_whole():
    df = get_hydro_production()
    assert isinstance(df, pd.DataFrame)
