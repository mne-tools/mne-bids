import json
from jsonschema import validate
import os.path as op

from mne.datasets import sample

data_path = sample.data_path()
json_fname = op.join('.', 'sub-01_task-audiovisual_meg.json')

empty = {"type": "string", "maxLength": 0}

# A sample schema, like what we'd get from json.load()
schema = {
    "type": "object",
    "properties": {
            "TaskName": {"type": "string", "minLength": 1},
            "SamplingFrequency": {"type": "number"},
            "Manufacturer": {"type": "string", "minLength": 1},
            "ManufacturerModelName": {"anyOf": [{"type": "string"}, empty]},
    },
}

with open(json_fname) as json_data:
    meg_json = json.load(json_data)

validate(meg_json, schema)
