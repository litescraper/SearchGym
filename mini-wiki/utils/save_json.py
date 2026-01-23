# Read json from model output and save to json file

import json

def save_json(json_data: dict, save_path: str):
    with open(save_path, "w") as f:
        json.dump(json_data, f)