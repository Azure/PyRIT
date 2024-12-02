# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import yaml


def try_read_key(key):
    try:
        with open(os.path.expanduser(key)) as infile:
            return infile.read().strip()
    except FileNotFoundError:
        return key


def update(d, u):
    """
    Recursive update the "base" dictionary `d` with values from `u`
    Modified from
    https://stackoverflow.com/a/3233356/5712749
    """
    for k, v in u.items():
        if k not in d:
            raise KeyError(f"parameter {k} not in `_default.toml`")
        if isinstance(v, dict):
            d[k] = update(d[k], v)
        else:
            d[k] = v
    return d


def load_config(default_fpath="configs/_default.toml"):
    """
    Load the configuration YAML into a dictionary.

    `default_fpath` is a file where all settings are specified.
    """
    with open(default_fpath) as infile:
        config = yaml.safe_load(infile)

    config["openai_api_key"] = os.getenv("OPENAI_API_KEY") or try_read_key(config["openai_api_key"])
    config["perspective_api_key"] = try_read_key(config["perspective_api_key"])
    return config


if __name__ == "__main__":
    config = load_config("configs/_default.yaml")
    print(config)
