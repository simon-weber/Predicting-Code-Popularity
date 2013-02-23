"""Project-global configuration."""

import json

config_file = 'config.json'

with open(config_file, 'rb') as f:
    config = json.load(f)

repo_dir = 'repos'  # format repo_dir/user/repo
feature_pickle_name = 'features.pickle'
