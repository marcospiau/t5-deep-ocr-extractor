import yaml


# main config file, should be in root path
MAIN_CONFIG_FILE="./config.yml"

def parse_yaml_file(x):
    with open(x, 'r') as f:
        parsed_yaml_file = yaml.load(f, Loader=yaml.FullLoader)
    return parsed_yaml_file

def load_main_config():
    return parse_yaml_file(MAIN_CONFIG_FILE)


