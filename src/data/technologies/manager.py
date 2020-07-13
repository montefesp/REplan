from os.path import join, dirname, abspath
from typing import List, Dict, Any, Union
import yaml
import pandas as pd


def get_config_dict(tech_names: List[str] = None, params: Union[List[str], List[List[str]]] = None) -> Dict[str, Any]:
    """
    Returns a dictionary associating to each technology name a list of required configuration parameters.

    Parameters
    ----------
    tech_names: List[str]
        Technology names.
    params: Union[List[str], List[List[str]]] (default: None)
        Names of configuration parameters.
        If None returns all available parameters for each technology.
        If List[str], returns the same parameters for each technology name.
        If List[List[str]], returns the independent parameters for each technology name.

    Returns
    -------
    tech_conf: Dict[str, Any]
        Dictionary associating to each technology name the required configuration parameters.

    """

    if tech_names is not None:
        assert len(tech_names) != 0, "Error: List of technology names is empty."
    if params is not None:
        assert len(params) != 0, "Error: List of parameters is empty."

    tech_conf_path = join(dirname(abspath(__file__)), '../../../data/technologies/tech_config.yml')
    tech_conf_all = yaml.load(open(tech_conf_path, 'r'), Loader=yaml.FullLoader)

    if tech_names is None:
        tech_names = list(tech_conf_all.keys())

    tech_conf = {}
    for tech_name in tech_names:
        assert tech_name in tech_conf_all, f"Error: No configuration is written for technology name {tech_name}."
        if params is not None:
            tech_conf[tech_name] = {}
            for param in params:
                assert param in tech_conf_all[tech_name],\
                    f"Error: Parameter {param} is not defined for technology name {tech_name}."
                tech_conf[tech_name][param] = tech_conf_all[tech_name][param]
        else:
            tech_conf[tech_name] = tech_conf_all[tech_name]

    return tech_conf


def get_config_values(tech_name: str, params: List[str]) -> Union[Any, List[Any]]:
    """
    Return the values corresponding to a series of configuration parameters.

    Parameters
    ----------
    tech_name: str
        Technology name.
    params: List[str]
        List of parameters.
    Returns
    -------
    Unique or list of required parameters values.

    """
    assert len(params) != 0, "Error: List of parameters is empty"

    tech_config_dict = get_config_dict([tech_name], params)
    if len(params) == 1:
        return tech_config_dict[tech_name][params[0]]
    else:
        return [tech_config_dict[tech_name][param] for param in params]


def get_info(tech_name: str, params: List[str]) -> pd.Series:
    """
    Return some information about a pre-defined technology.

    Parameters
    ----------
    tech_name: str
        Technology name.
    params: List[str]
        Name of parameters.

    Returns
    -------
    pd.Series
        Series containing the value for each parameter.

    """
    assert len(params) != 0, "Error: List of parameters is empty."

    tech_info_fn = join(dirname(abspath(__file__)), "../../../data/technologies/tech_info.xlsx")
    plant, plant_type = get_config_values(tech_name, ["plant", "type"])
    tech_info = pd.read_excel(tech_info_fn, sheet_name='values', index_col=[0, 1]).loc[plant, plant_type]

    for param in params:
        assert param in tech_info, f"Error: There is no parameter {param} for technology {tech_name}."

    return tech_info[params]
