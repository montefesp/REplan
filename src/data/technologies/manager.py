from os.path import join, dirname, abspath
from typing import Tuple
import yaml


def get_plant_type(tech: str) -> Tuple[str, str]:

    tech_conf_path = join(dirname(abspath(__file__)), '../../../data/technologies/tech_config.yml')
    tech_conf = yaml.load(open(tech_conf_path, 'r'), Loader=yaml.FullLoader)

    assert tech in tech_conf, f"Error: Technology {tech} configuration is not defined."
    for key in ['plant', 'type']:
        assert key in tech_conf[tech], f"Error: {key} undefined for technology {tech}"

    return tech_conf[tech]['plant'], tech_conf[tech]['type']


