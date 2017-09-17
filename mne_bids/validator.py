import os.path as op
import inspect

import json
from jsonschema import validate

FILE = inspect.getfile(inspect.currentframe())
base_dir = op.join(op.dirname(op.abspath(FILE)), 'jsonschemas')


def validate_meg(fname):
    # get the correct schema
    if fname.endswith('_meg.json'):
        schema_fname = 'schema_meg.json'
    elif fname.endswith('_fid.json'):
        schema_fname = 'schema_meg_fid.json'

    # open the schema
    with open(op.join(base_dir, schema_fname)) as json_data:
        schema = json.load(json_data)

    # open the BIDS json file to validate
    with open(fname) as json_data:
        test_json = json.load(json_data)

    # validate it
    validate(test_json, schema)
