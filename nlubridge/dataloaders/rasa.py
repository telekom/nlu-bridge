# Copyright (c) 2021 Ralf Kirchherr, Deutsche Telekom AG
# This software is distributed under the terms of the MIT license
# which is available at https://opensource.org/licenses/MIT

from __future__ import annotations

import pathlib
from typing import Dict, List, Union

from rasa.shared.data import get_data_files, is_nlu_file
from rasa.shared.importers.utils import training_data_from_paths
from rasa.shared.nlu.constants import (
    ENTITIES,
    ENTITY_ATTRIBUTE_END,
    ENTITY_ATTRIBUTE_START,
    ENTITY_ATTRIBUTE_TYPE,
    ENTITY_ATTRIBUTE_VALUE,
    INTENT,
    TEXT,
)
from rasa.shared.nlu.training_data.formats.rasa import RasaReader, RasaWriter
from rasa.shared.nlu.training_data.formats.rasa_yaml import RasaYAMLWriter
from rasa.shared.nlu.training_data.message import Message
from rasa.shared.nlu.training_data.training_data import TrainingData
from rasa.shared.utils.io import write_yaml

from nlubridge.nlu_dataset import EntityKeys, NluDataset


def from_rasa(
    filepath: Union[str, pathlib.Path], format: str = "yml", verbose=True
) -> NluDataset:
    """
    Load data stored in Rasa's yml- resp. json-format as NLUdataset.

    :param filepath: file path to read data from (Rasa specific yml- or json-format).
                     If 'yml' format is selected, this can be a directory with several
                     files.
    :param format: Input format, 'yml' (default) or 'json'
    :return: The loaded dataset as NLUdataset object
    """
    if format == "yml":
        training_files = get_data_files([filepath], is_nlu_file)
        # TODO: Replace with log messages!
        if verbose:
            print(f"Found {len(training_files)} NLU files")
            print("Reading data...")
        # Language tag in this function seems to be meaningless as in Rasa source code
        # apparently is never set to anything else than default "en"
        training_data = training_data_from_paths(training_files, "en")
    elif format == "json":
        training_data = RasaReader().read(filename=filepath)
    else:
        raise ValueError(f"Unknown format {format!r}")

    texts = []
    intents = []
    entities = []
    for message in training_data.training_examples:
        texts.append(message.get(TEXT))
        intents.append(message.get(INTENT))
        es = convert_entities_to_nludataset(message)
        entities.append(es)

    return NluDataset(texts, intents, entities)


def convert_entities_to_nludataset(message: Message) -> List[Dict]:
    """
    Convert entity dicts from Rasa format to NluDataset format.

    It will change the key names for entity type, start index and end index to those
    used in NluDataset and keep all additional keys from the Rasa formatted entity.

    :param message: A Rasa Message object
    :return: list of entities (dicts) in new format
    """
    es = []
    for e in message.get(ENTITIES, []):
        entity = {
            EntityKeys.TYPE: e.get(ENTITY_ATTRIBUTE_TYPE),
            EntityKeys.START: e.get(ENTITY_ATTRIBUTE_START),
            EntityKeys.END: e.get(ENTITY_ATTRIBUTE_END),
        }
        # Add any custom keys defined in the source structure
        for key in e.keys():
            if key not in [
                ENTITY_ATTRIBUTE_TYPE,
                ENTITY_ATTRIBUTE_START,
                ENTITY_ATTRIBUTE_END,
            ]:
                entity[key] = e[key]
        es.append(entity)
    return es


def to_rasa(
    dataset: NluDataset, filepath: Union[str, pathlib.Path], format: str = "yml"
):
    """
    Write dataset in Rasa's yml- or json-format.

    :param dataset: Dataset to be converted
    :param filepath: Path of the output yml file
    :param format: Output format, 'yml' (default) or 'json'
    """
    # TODO: This method should share code with the Rasa._convert() method
    messages = []
    for text, intent, entities in zip(dataset.texts, dataset.intents, dataset.entities):
        formatted_entities = []
        for e in entities:
            formatted_entity = {
                ENTITY_ATTRIBUTE_TYPE: e[EntityKeys.TYPE],
                ENTITY_ATTRIBUTE_START: e[EntityKeys.START],
                ENTITY_ATTRIBUTE_END: e[EntityKeys.END],
                # Please note: This sets just the default 'value' (if the input dataset
                # provides an explicit 'value' parameter, it will be adapted accordingly
                # in the section for custom keys below)
                ENTITY_ATTRIBUTE_VALUE: text[e[EntityKeys.START] : e[EntityKeys.END]],
            }
            # Add any custom keys defined in the source structure
            for key in e.keys():
                if key not in [EntityKeys.TYPE, EntityKeys.START, EntityKeys.END]:
                    formatted_entity[key] = e[key]
            formatted_entities.append(formatted_entity)
        example = {
            TEXT: text,
            INTENT: intent if intent is not None else "default_intent",
            ENTITIES: formatted_entities,
        }
        message = Message(data=example)
        messages.append(message)

    training_data = TrainingData(training_examples=messages)
    if format == "yml":
        mry = RasaYAMLWriter()
        md = mry.training_data_to_dict(training_data)
        write_yaml(md, filepath)
    elif format == "json":
        mrj = RasaWriter()
        mrj.dump(filepath, training_data)
    else:
        raise ValueError(f"Unsupported format {format!r}")
