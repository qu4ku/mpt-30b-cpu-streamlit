import os
import json

from huggingface_hub import hf_hub_download


def parse_json_config(file_path):
    with open(file_path, 'r') as json_file:
        data = json.load(json_file)
    return data

