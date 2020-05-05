from typing import Tuple


# TODO: - This function allows to do the conversion to David's file until we homogenize techs
#       - Should be defined in some files
def get_plant_type(tech: str) -> Tuple[str, str]:

    if tech == "ccgt":
        return "NGPP", "CCGT"
    elif tech == "ocgt":
        return "NGPP", "OCGT"
    elif tech == "nuclear":
        return "Nuclear", "Uranium"
    elif tech == "sto":
        return "Hydro", "Reservoir"
    elif tech == "ror":
        return "Hydro", "Run-of-river"
    elif tech == "phs":
        return "Storage", "Pumped-hydro"
    elif tech == "Li-ion":
        return "Storage", "Li-ion"
    elif tech == "wind_onshore":
        return "Wind", "Onshore"
    elif tech == "wind_offshore":
        return "Wind", "Offshore"
    elif tech == "wind_floating":
        return "Wind", "Floating"
    elif tech == "pv_utility":
        return "PV", "Utility"
    elif tech == "pv_residential":
        return "PV", "Residential"
    # TODO: For now consider overhead lines for HVAC and undersea cables for HVDC
    #  Would need to do sht much more clever
    elif tech == "AC":
        return "Transmission", "HVAC_OHL"
    elif tech == "DC":
        return "Transmission", "HVDC_SC"
    else:
        raise ValueError(f"No available plant technology and type for {tech}")
