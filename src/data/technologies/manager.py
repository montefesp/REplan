from os.path import join, dirname, abspath
from typing import Tuple, List, Dict, Any, Union
import yaml


# TODO: this will probably be replaced by the following function
def get_plant_type(tech_name: str) -> Tuple[str, str]:

    tech_conf_path = join(dirname(abspath(__file__)), '../../../data/technologies/tech_config.yml')
    tech_conf = yaml.load(open(tech_conf_path, 'r'), Loader=yaml.FullLoader)

    assert tech_name in tech_conf, f"Error: Technology {tech_name} configuration is not defined."
    for key in ['plant', 'type']:
        assert key in tech_conf[tech_name], f"Error: {key} undefined for technology {tech_name}"

    return tech_conf[tech_name]['plant'], tech_conf[tech_name]['type']


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


# TODO: comment and test
def get_config_values(tech_name: str, params: List[str]) -> Union[Any, List[Any]]:

    assert len(params) != 0, "Error: List of parameters is empty"

    tech_config_dict = get_config_dict([tech_name], params)
    if len(params) == 1:
        return tech_config_dict[tech_name][params[0]]
    else:
        return [tech_config_dict[tech_name][param] for param in params]
