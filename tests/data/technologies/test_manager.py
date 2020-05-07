import pytest

from src.data.technologies.costs import *


def test_wrong_tech():
    with pytest.raises(ValueError):
        get_plant_type('ABC')