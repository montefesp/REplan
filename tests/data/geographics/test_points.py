import pytest

from src.data.geographics.points import *
from src.data.geographics import get_shapes


def test_match_points_to_regions_empty_list_of_points():

    onshore_shapes = get_shapes(["BE"], "onshore")["geometry"]
    with pytest.raises(AssertionError):
        match_points_to_regions([], onshore_shapes)


def test_match_points_to_regions_empty_list_of_regions():

    onshore_shapes = pd.Series()
    with pytest.raises(AssertionError):
        match_points_to_regions([(4.3053506, 50.8550625)], onshore_shapes)


def test_match_points_to_regions_one_point_in_one_shape():

    onshore_shapes = get_shapes(["BE"], "onshore")["geometry"]
    ds = match_points_to_regions([(4.3053506, 50.8550625)], onshore_shapes, keep_outside=False)
    assert isinstance(ds, pd.Series)
    assert(len(ds) == 1)
    assert (4.3053506, 50.8550625) in ds.index
    assert ds[(4.3053506, 50.8550625)] == "BE"


def test_match_points_to_regions_one_point_in_two_shapes():

    onshore_shapes = get_shapes(["BE", "NL"], "onshore")["geometry"]
    ds = match_points_to_regions([(4.3053506, 50.8550625)], onshore_shapes, keep_outside=False)
    assert isinstance(ds, pd.Series)
    assert(len(ds) == 1)
    assert (4.3053506, 50.8550625) in ds.index
    assert ds[(4.3053506, 50.8550625)] == "BE"


def test_match_points_to_regions_two_point_in_two_shapes():

    onshore_shapes = get_shapes(["BE", "NL"], "onshore")["geometry"]
    ds = match_points_to_regions([(4.3053506, 50.8550625), (4.8339211, 52.3547498)], onshore_shapes, keep_outside=False)
    assert isinstance(ds, pd.Series)
    assert(len(ds) == 2)
    assert (4.3053506, 50.8550625) in ds.index
    assert (4.8339211, 52.3547498) in ds.index
    assert ds[(4.3053506, 50.8550625)] == "BE"
    assert ds[(4.8339211, 52.3547498)] == "NL"


def test_match_points_to_regions_one_point_near_shape_not_keeping():

    onshore_shapes = get_shapes(["NL"], "onshore")["geometry"]
    ds = match_points_to_regions([(3.9855853, 51.9205033)], onshore_shapes, keep_outside=False)
    assert isinstance(ds, pd.Series)
    assert(len(ds) == 1)
    assert (3.9855853, 51.9205033) in ds.index
    assert np.isnan(ds[(3.9855853, 51.9205033)])


def test_match_points_to_regions_one_point_near_shape_keeping():

    onshore_shapes = get_shapes(["NL", "BE"], "onshore")["geometry"]
    ds = match_points_to_regions([(3.9855853, 51.9205033)], onshore_shapes, distance_threshold=5.)
    assert isinstance(ds, pd.Series)
    assert(len(ds) == 1)
    assert (3.9855853, 51.9205033) in ds.index
    assert ds[(3.9855853, 51.9205033)] == "NL"


def test_match_points_to_regions_one_point_away_from_shape_keeping():

    onshore_shapes = get_shapes(["NL", "BE"], "onshore")["geometry"]
    ds = match_points_to_regions([(3.91953, 52.0067)], onshore_shapes, distance_threshold=5.)
    assert isinstance(ds, pd.Series)
    assert (len(ds) == 1)
    assert (3.91953, 52.0067) in ds.index
    assert np.isnan(ds[(3.91953, 52.0067)])


def test_match_points_to_countries_empty_list_of_points():
    with pytest.raises(AssertionError):
        match_points_to_countries([], ["BE"])


def test_match_points_to_countries_empty_list_of_countries():
    with pytest.raises(AssertionError):
        match_points_to_countries([(4.3053506, 50.8550625)], [])


def test_match_points_to_countries_wrong_countries():
    with pytest.raises(AssertionError):
        match_points_to_countries([(4.3053506, 50.8550625)], ["ZZ"])


def test_get_points_in_shape_too_coarse_resolution():
    shape = get_shapes(["BE"], "onshore").loc["BE", "geometry"]
    res = 10.0
    points = get_points_in_shape(shape, res)
    assert isinstance(points, list)
    assert len(points) == 0


def test_get_points_in_shape_without_init_points():
    shape = get_shapes(["BE"], "onshore").loc["BE", "geometry"]
    res = 0.5
    points = get_points_in_shape(shape, res)
    assert isinstance(points, list)
    assert all(isinstance(point, tuple) for point in points)
    assert all(len(point) == 2 for point in points)
    assert all(map(lambda point: int(point[0] / res) == point[0] / res and
               int(point[1] / res) == point[1] / res, points))


def test_get_points_in_shape_with_init_points():
    shape = get_shapes(["BE"], "onshore").loc["BE", "geometry"]
    res = 1.0
    points_in = [(4.0, 51.0), (5.0, 50.0)]
    point_out = [(4.0, 52.0), (3.0, 50.0)]
    init_points = point_out + points_in
    points = get_points_in_shape(shape, res, init_points)
    assert isinstance(points, list)
    assert all(isinstance(point, tuple) for point in points)
    assert all(len(point) == 2 for point in points)
    assert all(map(lambda point: int(point[0] / res) == point[0] / res and
               int(point[1] / res) == point[1] / res, points))
    assert points == points_in
