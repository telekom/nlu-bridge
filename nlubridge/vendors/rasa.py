# Copyright (c) 2021 Klaus-Peter Engelbrecht, Deutsche Telekom AG
# This software is distributed under the terms of the MIT license
# which is available at https://opensource.org/licenses/MIT
from __future__ import annotations
from typing import List, Optional, Union, Tuple
import os
import pathlib

from rasa.nlu import config
from rasa.nlu.model import Trainer
from rasa.shared.nlu.training_data.formats.rasa import RasaReader
from rasa.shared.nlu.training_data.training_data import TrainingData
from rasa.shared.nlu.constants import (
    TEXT,
    INTENT,
    INTENT_NAME_KEY,
    ENTITIES,
    ENTITY_ATTRIBUTE_TYPE,
    ENTITY_ATTRIBUTE_START,
    ENTITY_ATTRIBUTE_END,
    ENTITY_ATTRIBUTE_VALUE,
    PREDICTED_CONFIDENCE_KEY,
)

from .vendors import Vendor
from nlubridge.datasets import NLUdataset, EntityKeys


DEFAULT_INTENT_RASA_CONFIG_PATH = os.path.join(
    pathlib.Path(__file__).parent.absolute(), "config", "rasa_nlu_config.yml"
)
ENTITY_KEY_VALUE = "value"  # Rasa provides an explicit value parameter for its entities


class Rasa(Vendor):
    alias = "rasa"

    def __init__(self, model_config: Optional[str] = None):
        """
        Interface for the `Rasa NLU <https://github.com/RasaHQ/rasa>`_.

        Uses the default pipeline as of January 22 2021. No algorithmic
        details have been researched. The configuration file can be
        found in the config directory within this directory. A custom
        pipeline can be provided as argument to the class constructor.

        :param model_config: filepath to a Rasa config file
        """
        self.config = model_config
        self.interpreter = None

    def train(self, dataset: NLUdataset) -> Rasa:
        """
        Train intent and/or entity classification.

        :param dataset: Training data
        :return: It's own Rasa object
        """
        self.config = self.config if self.config else DEFAULT_INTENT_RASA_CONFIG_PATH
        training_data = self._convert(dataset)
        trainer = Trainer(config.load(self.config))
        self.interpreter = trainer.train(training_data)
        return self

    def train_intent(self, dataset: NLUdataset) -> Rasa:
        """
        Train intent classification.

        This method is mainly for compatibility reasons, as it in case of Rasa identical
        to the `train` method.

        :param dataset: Training data
        :return: It's own Rasa object
        """
        return self.train(dataset)

    def test(self, dataset: NLUdataset) -> NLUdataset:
        """
        Test a given dataset.

        Test a given dataset and obtain the intent and/or entity classification results
        in the NLUdataset format.

        :param dataset: Input dataset to be tested
        :return: NLUdataset object comprising the classification results. The list of
            the predicted intent classification probabilities are accessible via the
            additional attribute 'probs' (List[float]).
        """
        intents = []
        probs = []
        entities_list = []
        for text in dataset.texts:
            result = self.interpreter.parse(text)
            intent = result.get(INTENT, {}).get(INTENT_NAME_KEY)
            prob = result.get(INTENT, {}).get(PREDICTED_CONFIDENCE_KEY)
            entities = [
                {
                    EntityKeys.TYPE: e.get(ENTITY_ATTRIBUTE_TYPE),
                    EntityKeys.START: e.get(ENTITY_ATTRIBUTE_START),
                    EntityKeys.END: e.get(ENTITY_ATTRIBUTE_END),
                }
                for e in result.get(ENTITIES, [])
            ]

            intents.append(intent)
            probs.append(prob)
            entities_list.append(entities)

        res = NLUdataset(dataset.texts, intents, entities_list)
        res.probs = probs
        return res

    def test_intent(
        self, dataset: NLUdataset, return_probs: bool = False
    ) -> Union[List[str], Tuple[List[str], List[float]]]:
        """
        Test a given dataset and obtain just the intent classification results.

        :param dataset: The dataset to be tested
        :param return_probs: Specifies if the probability values should be returned
            (default is False)
        :return: Either a list of predicted intent classification results or a tuple of
            predicted intent classification and probabilites results (depeding on
            argument 'return_probs')
        """
        intents = []
        probs = []
        for text in dataset.texts:
            result = self.interpreter.parse(text)
            intent = result.get(INTENT, {}).get(INTENT_NAME_KEY)
            prob = result.get(INTENT, {}).get(PREDICTED_CONFIDENCE_KEY)
            intents.append(intent)
            probs.append(prob)
        if return_probs:
            return intents, probs
        return intents

    @staticmethod
    def _convert(dataset: NLUdataset) -> TrainingData:
        """
        Convert a NLUdataset to a Rasa TrainingData object.

        :param dataset: NLUdataset to be converted
        :return: Rasa TrainingData object
        """
        examples = []

        for text, intent, entities in dataset:
            example = {
                TEXT: text,
                INTENT: intent if intent is not None else "default_intent",
                ENTITIES: [],
            }
            for entity in entities:
                formatted_entity = {
                    ENTITY_ATTRIBUTE_TYPE: entity[EntityKeys.TYPE],
                    ENTITY_ATTRIBUTE_START: entity[EntityKeys.START],
                    ENTITY_ATTRIBUTE_END: entity[EntityKeys.END],
                    # Please note: This sets just the default 'value' (if the input
                    # dataset provides an explicit 'value' parameter, it will be adapted
                    # accordingly in the section for custom keys below)
                    ENTITY_ATTRIBUTE_VALUE: text[
                        entity[EntityKeys.START] : entity[EntityKeys.END]
                    ],
                }
                # Add any custom keys defined in the source structure
                for key in entity.keys():
                    if key not in [EntityKeys.TYPE, EntityKeys.START, EntityKeys.END]:
                        formatted_entity[key] = entity[key]
                example[ENTITIES].append(formatted_entity)
            examples.append(example)

        training_data = {
            "rasa_nlu_data": {
                "common_examples": examples,
                "regex_features": [],
                "entity_synonyms": [],
            }
        }

        return RasaReader().read_from_json(training_data)
