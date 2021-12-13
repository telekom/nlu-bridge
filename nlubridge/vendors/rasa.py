# Copyright (c) 2021 Klaus-Peter Engelbrecht, Deutsche Telekom AG
# This software is distributed under the terms of the MIT license
# which is available at https://opensource.org/licenses/MIT

import os
import pathlib
import json

from lazy_imports import try_import

with try_import() as optional_rasa_import:
    from rasa.nlu import config
    from rasa.nlu.model import Trainer
    from rasa.shared.nlu.training_data.formats.rasa import RasaReader

from .vendors import Vendor
from nlubridge.datasets import from_json, NLUdataset


DEFAULT_RASA_CONFIG_PATH = os.path.join(
    pathlib.Path(__file__).parent.absolute(), "config", "rasa_nlu_config.yml"
)


class Rasa(Vendor):
    alias = "rasa"

    def __init__(self, model_config=DEFAULT_RASA_CONFIG_PATH):
        """
        Interface for the `Rasa NLU <https://github.com/RasaHQ/rasa>`_.

        Uses the default pipeline as of January 22 2021. No algorithmic
        details have been researched. The configuration file can be
        found in the config directory within this directory. A custom
        pipeline can be provided as argument to the class constructor.

        :param model_config: filepath to a Rasa config file
        :type model_config: str
        """
        optional_rasa_import.check()
        self.config = config.load(model_config)
        self.interpreter = None

    def train_intent(self, dataset):
        """Train intent classifier."""
        training_data = self._convert(dataset)
        trainer = Trainer(self.config)
        self.interpreter = trainer.train(training_data)
        return self

    def test_intent(self, dataset, return_probs=False):
        """Test intent classifier."""
        intents = []
        probs = []
        for text in dataset.texts:
            result = self.interpreter.parse(text)
            intent = result["intent"]["name"]
            prob = result["intent"]["confidence"]
            intents.append(intent)
            probs.append(prob)
        if return_probs:
            return intents, probs
        return intents

    @staticmethod
    def _convert(dataset):
        """Convert a NLUdataset to a Rasa TrainingData object."""
        examples = []

        for text, intent, entities in dataset:
            example = {
                "text": text,
                "intent": intent,
                "entities": [
                    {
                        "value": entity["value"],
                        "entity": entity["entity"],
                        "start": entity["start"],
                        "end": entity["end"],
                    }
                    for entity in entities
                ],
            }  # yapf: disable
            examples.append(example)

        training_data = {
            "rasa_nlu_data": {
                "common_examples": examples,
                "regex_features": [],
                "entity_synonyms": [],
            }
        }

        return RasaReader().read_from_json(training_data)


def load_data(filepath, format="json") -> NLUdataset:
    """
    Load data stored in Rasa format as NLUdataset.

    :param filepath: file path to read data from
    :type filepath: str
    :param format: "json" means data under filepath are stored in
        Rasa's JSON format. Alternatively, "md" indicates the new
        markdown format (not yet supported)
    :type format: str
    """
    if format != "json":
        raise NotImplementedError("Can only read JSON format for Rasa")
    with open(filepath, "r") as f:
        examples = json.load(f)["rasa_nlu_data"]["common_examples"]
    dataset = from_json(
        json.dumps(examples, ensure_ascii=False),
        text_key="text",
        intent_key="intent",
        entities_key="entities",
        entity_type_key="entity",
        entity_start_key="start",
        entity_end_key="end",
        end_index_add_1=False,
    )
    return dataset


def write_data(dataset, filepath):
    """Write dataset in Rasa's JSON format."""
    records = dataset.to_records()
    rasa_template = {
        "rasa_nlu_data": {
            "common_examples": records,
            "regex_features": [],
            "entity_synonyms": [],
        }
    }
    dataset.to_json(filepath, rasa_template)
