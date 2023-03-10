# Copyright (c) 2021 Klaus-Peter Engelbrecht, Deutsche Telekom AG
# This software is distributed under the terms of the MIT license
# which is available at https://opensource.org/licenses/MIT

from nlubridge.nlu_dataset import NluDataset

from .utils import from_json


def from_luis(filepath) -> NluDataset:
    """
    Load data in LUIS format as NLUdataset.

    :param filepath: file path to the LUIS-formatted data.
    :type filepath: str
    """
    dataset = from_json(
        path=filepath,
        text_key="text",
        intent_key="intentName",
        entities_key="entityLabels",
        entity_type_key="entityName",
        entity_start_key="startCharIndex",
        entity_end_key="endCharIndex",
        end_index_add_1=True,
    )
    return dataset
