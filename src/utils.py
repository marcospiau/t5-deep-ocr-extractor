import yaml


def parse_yaml_file(x):
    with open(x, 'r') as f:
        parsed_yaml_file = yaml.load(f, Loader=yaml.FullLoader)
    return parsed_yaml_file
