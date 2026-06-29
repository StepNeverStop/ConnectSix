"""Configuration loading utilities."""

import os

import yaml


def load_config(filename):
    if not os.path.exists(filename):
        raise FileNotFoundError(f'cannot find config file: {filename}')
    with open(filename, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f.read())
