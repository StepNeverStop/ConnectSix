import os
import yaml

def load_config(filename):
    if os.path.exists(filename):
        with open(filename, 'r', encoding='utf-8') as f:
            x = yaml.safe_load(f.read())
    else:
        raise Exception('cannot find this config.')
    return x