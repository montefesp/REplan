import pytest

from src.resite.grid_cells import *
from src.data.geographics import get_shapes


def test_create_grid_cells_too_coarse_resolution():
    shape = get_shapes(["BE"], "onshore").loc["BE", "geometry"]
    res = 10.0
    points, gc = create_grid_cells(shape, res)
    assert isinstance(points, list)
    assert isinstance(gc, list)
    assert len(points) == 0
    assert len(gc) == 0


def test_create_grid_cells():
    shape = get_shapes(["BE"], "onshore").loc["BE", "geometry"]
    res = 1.0
    points, gc = create_grid_cells(shape, res)
    assert len(gc) == len(points)
    assert all([isinstance(cell, Polygon) or isinstance(cell, MultiPolygon) for cell in gc])
    areas_sum = sum([cell.area for cell in gc])
    assert abs(areas_sum - shape.area)/max(areas_sum, shape.area) < 0.01


def get_tech_config():
    return {'wind_onshore': {'onshore': True},
            'wind_offshore': {'onshore': False},
            'pv_utility': {'onshore': True}}


def test_get_grid_cells_empty_list_of_technologies():
    with pytest.raises(AssertionError):
        get_grid_cells([], 0.5)


def test_get_grid_cells_missing_shapes():
    shapes = get_shapes(["BE"])
    onshore_shape = shapes[~shapes["offshore"]].loc["BE", "geometry"]
    offshore_shape = shapes[shapes["offshore"]].loc["BE", "geometry"]
    with pytest.raises(AssertionError):
        get_grid_cells(['wind_onshore'], 0.5, offshore_shape=offshore_shape)
    with pytest.raises(AssertionError):
        get_grid_cells(['wind_offshore'], 0.5, onshore_shape=onshore_shape)


def test_get_grid_cells():
    shapes = get_shapes(["BE"])
    onshore_shape = shapes[~shapes["offshore"]].loc["BE", "geometry"]
    offshore_shape = shapes[shapes["offshore"]].loc["BE", "geometry"]
    ds = get_grid_cells(['wind_onshore', 'wind_offshore', 'pv_utility'],
                        0.25, onshore_shape, offshore_shape)

    assert isinstance(ds, pd.Series)
    assert len(ds['wind_offshore']) == 6
    assert len(ds['wind_onshore']) == 61
    assert len(ds['pv_utility']) == 61
    assert (ds['wind_onshore'] == ds['pv_utility']).all()
