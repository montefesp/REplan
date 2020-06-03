import pytest

from src.resite.resite2 import Resite


def test_():
    resite = Resite(["BENELUX"], ["wind_onshore"], ['2015-01-01T00:00', '2015-01-01T23:00'], 0.5)
    resite.build_input_data(False)
    resite.build_model("docplex", "meet_demand_with_capacity", {"cap_per_tech": [1]})
    resite.solve_model()
    print(resite.retrieve_solution())
