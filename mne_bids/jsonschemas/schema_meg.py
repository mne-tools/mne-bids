from jsonschema import validate
import numpy as np

empty = { "type": "string", "maxLength": 0}

NaN = np.nan;
# booleans have problems with capitals (e.g. True versus true), this is a string for now.

# A sample MEG schema, like what we'd get from json.load()
schema = {
        "type": "object",
        "properties": {
                "TaskName": {"type": "string","minLength": 1},
                "Manufacturer": {"type": "string","minLength": 1},
                "ManufacturerModelName": {"type": "string"},
                "TaskDescription": {"type": "string"},
                "Instructions": {"type": "string"},
                "PowerLineFrequency": {"type": "number"},
                "CogAtlasID": {"type": "string"},
                "CogPOID": {"type": "string"},
                "InstitutionName": {"type": "string"},
                "InstitutionAddress": {"type": "string"},
                "DeviceSerialNumber": {"type": "string"},
                "MEGChannelCount": {"type": "integer"},
                "MEGREFChannelCount": {"type": "integer"},
                "EEGChannelCount": {"type" : "integer"},
                "EOGChannelCount": {"type" : "integer"},
                "ECGChannelCount": {"type" : "integer"},
                "EMGChannelCount": {"type" : "integer"},
                "MiscChannelCount": {"type" : "integer"},
                "TriggerChannelCount": {"type" : "integer"},
                "PowerLineFrequency": {"type" : "number"},
                "EEGPlacementScheme": {"type" : "string"},
                "EEGReference": {"type" : "string"},
                "DewarPosition": {"type" : "string"},
                "SoftwareFilters": {"type" : "string"},
                "RecordingDuration": {"type" : "number"},
                "RecordingType": {"type" : "string"},
                "EpochLength": {"type" : "number"},
                "DeviceSoftwareVersion": {"type" : "string"},
                "ContinuousHeadLocalization": {"type" : "string"},
                "CoilFrequency": {"type" : "number"},
                "MaxMovement": {"type" : "number"},
                "SubjectArtefactDescription": {"type" : "string"},
                "DigitizedLandmarks": {"type" : "string"},
                "DigitizedHeadPoints": {"type" : "string"},
                    },
        "required": [   "TaskName",
                        "Manufacturer",
                        "SamplingFrequency",
                        "MEGChannelCount",
                        "MEGREFChannelCount",
                        "EEGChannelCount",
                        "EOGChannelCount",
                        "ECGChannelCount",
                        "EMGChannelCount",
                        "MiscChannelCount",
                        "TriggerChannelCount"]
        }


test = {
    "TaskName": "audiovisual",
    "SamplingFrequency": 500,
    "Manufacturer": "test",
    "ManufacturerModelName": "test",
    "TaskDescription": "In this experiment...",
    "MEGChannelCount": 306,
    "MEGREFChannelCount": 0,
    "EEGChannelCount": 60,
    "EOGChannelCount": 1,
    "ECGChannelCount": 0,
    "EMGChannelCount": 0,
    "MiscChannelCount": 0,
    "TriggerChannelCount": 9,
    "PowerLineFrequency": NaN,
    "EEGReference": "nose",
    "MEGPosition": "upright",
    "OnlineFilters": "0.10000000149 Hz high-pass, 172.176300049 Hz low-pass",
    "RecordingType": "continuous",
    "EyesClosed": "false",
    "ContinuousHeadLocalization": "false",
    "DigitizedLandmarks": "true",
    "DigitizedHeadPoints": "true",
    "SubjectArtefactDescription": "",
}

validate(test, schema)
