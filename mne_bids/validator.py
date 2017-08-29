import os.path as op
import inspect

import json
from jsonschema import validate

FILE = inspect.getfile(inspect.currentframe())
base_dir = op.join(op.dirname(op.abspath(FILE)), 'jsonschemas')


def validate_meg(fname):
    with open(op.join(base_dir, 'schema_meg.json')) as json_data:
        schema = json.load(json_data)

    with open(fname) as json_data:
        meg_json = json.load(json_data)

    validate(meg_json, schema)
