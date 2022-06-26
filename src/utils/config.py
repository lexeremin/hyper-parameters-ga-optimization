import json
import os
from pathlib import Path


def get_project_config() -> Path:
    return os.path.join(Path(__file__).parent.parent, "configs/")


def get_config(fname):
    config = fname
    try:
        with open(os.path.join(get_project_config(), config)) as fd:
            data = json.load(fd)
    except IOError:
        print(f"Couldn't load file: {fname}")
        exit(0)
    return data


def get_ga_config():
    config = 'ga-config.json'
    return get_config(config)


def get_ts_data():
    config = 'data-config.json'
    return get_config(config)

# def get_hp_config():
#     config = 'hp-config.json'
#     return get_config(config)


def main():
    print(f'GA configuration: {get_ga_config()}')
    print(f'Data fetching configuration: {get_ts_data()}')
    # print(f'Hyperparameter configuration: {get_hp_config()}')


if __name__ == "__main__":
    main()
