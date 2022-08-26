# Copyright (c) 2021 Ralf Kirchherr, Deutsche Telekom AG
# This software is distributed under the terms of the MIT license
# which is available at https://opensource.org/licenses/MIT

from __future__ import annotations

import pathlib
from typing import Union

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
from rasa.shared.nlu.training_data.formats.rasa_yaml import (
    RasaYAMLReader,
    RasaYAMLWriter,
)
from rasa.shared.nlu.training_data.message import Message
from rasa.shared.nlu.training_data.training_data import TrainingData
from rasa.shared.utils.io import write_yaml

from nlubridge.nlu_dataset import EntityKeys, NluDataset


def from_rasa(filepath: Union[str, pathlib.Path], format: str = "yml") -> NluDataset:
    """
    Load data stored in Rasa's yml- resp. json-format as NLUdataset.

    :param filepath: file path to read data from (Rasa specific yml- or json-format)
    :param format: Input format, 'yml' (default) or 'json'
    :return: The loaded dataset as NLUdataset object
    """
    if format == "yml":
        trainingdata = RasaYAMLReader().read(filepath)
    elif format == "json":
        trainingdata = RasaReader().read(filename=filepath)
    else:
        raise ValueError(f"Unknown format {format!r}")

    texts = []
    intents = []
    entities = []
    for message in trainingdata.training_examples:
        texts.append(message.get(TEXT))
        intents.append(message.get(INTENT))
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
        entities.append(es)

    return NluDataset(texts, intents, entities)


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
    for text, intent, entities in dataset:
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
