# Copyright (c) 2022 Ralf Kirchherr, Deutsche Telekom AG
# This software is distributed under the terms of the MIT license
# which is available at https://opensource.org/licenses/MIT
from __future__ import annotations
import os
import logging
import asyncio
import tempfile
import pathlib
from typing import List, Optional, Union, Tuple, Dict

from lazy_imports import try_import

with try_import() as optional_rasa3_import:
    from rasa.model_training import train_nlu
    from rasa.core.agent import Agent
    from rasa.shared.nlu.training_data.training_data import TrainingData
    from rasa.shared.nlu.training_data.message import Message
    from rasa.shared.nlu.training_data.formats import rasa_yaml
    from rasa.shared.utils.io import write_yaml
    from rasa.shared.importers.rasa import RasaFileImporter
    from rasa.shared.importers.autoconfig import TrainingType
    from rasa.shared.nlu.constants import (
        TEXT,
        INTENT,
        INTENT_NAME_KEY,
        ENTITIES,
        ENTITY_ATTRIBUTE_TYPE,
        ENTITY_ATTRIBUTE_START,
        ENTITY_ATTRIBUTE_END,
        ENTITY_ATTRIBUTE_VALUE,
        PREDICTED_CONFIDENCE_KEY
    )

from .vendors import Vendor
from nlubridge.datasets import NLUdataset, EntityKeys

logger = logging.getLogger(__name__)

DEFAULT_INTENT_RASA_CONFIG_PATH = os.path.join(
    pathlib.Path(__file__).parent.absolute(), "config", "rasa_nlu_config.yml"
)
DEFAULT_ENTITY_RASA_CONFIG_PATH = os.path.join(
    pathlib.Path(__file__).parent.absolute(), "config", "rasa_nlu_entity_config.yml"
)


class Rasa3(Vendor):
    alias = "rasa3"

    def __init__(self, model_config: Optional[str] = None):
        """
        Interface for the `Rasa NLU <https://github.com/RasaHQ/rasa>`_.

        Uses the default pipeline as of January 22 2021. No algorithmic
        details have been researched. The configuration file can be
        found in the config directory within this directory. A custom
        pipeline can be provided as argument to the class constructor.

        :param model_config: filepath to a Rasa config file
        """
        optional_rasa3_import.check()
        self.config = model_config
        self.agent = None

    def train(self, dataset: NLUdataset) -> Rasa3:
        """
        Train intent and/or entity classification

        :param dataset: Training data
        :return: It's own Rasa3 object
        """
        self.config = self.config if self.config else DEFAULT_INTENT_RASA_CONFIG_PATH
        with tempfile.TemporaryDirectory() as tmpdirname:
            nlu_yml_file = os.path.join(pathlib.Path(tmpdirname), "nlu.yml")  # output path for temporary nlu.yml
            write_data(dataset, nlu_yml_file)
            logger.info(f"Start training (using {self.config!r})...")
            model_archive = train_nlu(self.config, nlu_yml_file, tmpdirname)
            logger.info(f"Training completed!")

            logger.info("Load model...")
            self.agent = Agent.load(model_path=model_archive)
            logger.info("Model loaded!")
        return self

    def train_intent(self, dataset: NLUdataset) -> Rasa3:
        """
        Train intent classification.
        This method is mainly for compatibility reasons, as it in case of Rasa identical to the `train` method.

        :param dataset: Training data
        :return: It's own Rasa3 object
       """
        return self.train(dataset)

    def train_entity(self, dataset: NLUdataset) -> Rasa3:
        """
        Train entity classification.
        This method is mainly for compatibility reasons. Only difference to the 'train' method is, that it uses a
        different default pipeline that does not expect any training data for intent classification.

        :param dataset: Training data
        :return: It's own Rasa3 object
       """

        self.config = self.config if self.config else DEFAULT_ENTITY_RASA_CONFIG_PATH
        return self.train(dataset)

    def test(self,
             dataset: NLUdataset
             ) -> NLUdataset:
        """
        Test a given dataset and obtain the intent and/or entity classification results in the NLUdataset format

        :param dataset: Input dataset to be tested
        :return: NLUdataset object comprising the classification results. The list of the predicted intent
                 classification probabilities are accessible via the additional attribute 'probs' (List[float]).
        """
        if self.agent is None:
            logger.error("Rasa3 classifier has to be trained first!")
        intents = []
        probs = []
        entities_list = []
        for text in dataset.texts:
            result = asyncio.run(self.agent.parse_message(text))  # agent's parse method is a coroutine
            intent = result.get(INTENT, {}).get(INTENT_NAME_KEY)
            prob = result.get(INTENT, {}).get(PREDICTED_CONFIDENCE_KEY)
            entities = [
                {
                    EntityKeys.TYPE: e.get(ENTITY_ATTRIBUTE_TYPE),
                    EntityKeys.START: e.get(ENTITY_ATTRIBUTE_START),
                    EntityKeys.END: e.get(ENTITY_ATTRIBUTE_END)
                } for e in result.get(ENTITIES, [])
            ]

            intents.append(intent)
            probs.append(prob)
            entities_list.append(entities)

        res = NLUdataset(dataset.texts, intents, entities_list)
        res.probs = probs
        return res

    def test_intent(self,
                    dataset: NLUdataset,
                    return_probs: bool = False
                    ) -> Union[List[str], Tuple[List[str], List[float]]]:
        """
        Test a given dataset and obtain just the intent classification results

        :param dataset: The dataset to be tested
        :param return_probs: Specifies if the probability values should be returned (default is False)
        :return: Either a list of predicted intent classification results or a tuple of predicted intent classification
                 and probabilites results (depeding on argument 'return_probs')
        """
        if self.agent is None:
            logger.error("Rasa3 classifier has to be trained first!")
        intents = []
        probs = []
        for text in dataset.texts:
            result = asyncio.run(self.agent.parse_message(text))  # agent's parse method is a coroutine
            intent = result.get(INTENT, {}).get(INTENT_NAME_KEY)
            prob = result.get(INTENT, {}).get(PREDICTED_CONFIDENCE_KEY)
            intents.append(intent)
            probs.append(prob)
        if return_probs:
            res = (intents, probs)
        else:
            res = intents
        return res

    def test_entity(self, dataset: NLUdataset) -> List[List[Dict]]:
        """
        Test a given dataset and obtain just the entity classification results.
        The entity classification results are returned in the original format of the Rasa classifier (i.e. including
        also the Rasa specific parameters).
        If the results should be compared with other classifiers it might be more convenient to use the 'test' method
        instead (which returns the result in a consistent format across all classifiers).

        :param dataset: The dataset to be tested
        :return: List of predicted entity results (Rasa entity format)
        """
        if self.agent is None:
            logger.error("Rasa3 classifier has to be trained first!")
        entities_list = []
        for text in dataset.texts:
            result = asyncio.run(self.agent.parse_message(text))  # agent's parse method is a coroutine
            entities = result[ENTITIES]
            entities_list.append(entities)
        return entities_list


def load_data(filepath: str) -> NLUdataset:
    """
    Load data stored in Rasa yml-format as NLUdataset.

    :param filepath: file path to read data from (Rasa specific yml format)
    :return: The loaded data set as NLUdataset object
    """
    importer = RasaFileImporter(training_data_paths=filepath, training_type=TrainingType.NLU)
    trainingdata = importer.get_nlu_data(language="de")
    texts = []
    intents = []
    entities = []
    for message in trainingdata.training_examples:
        texts.append(message.get(TEXT))
        intents.append(message.get(INTENT))
        es = []
        for e in message.get(ENTITIES, []):
            es.append(
                {
                    EntityKeys.TYPE: e.get(ENTITY_ATTRIBUTE_TYPE),
                    EntityKeys.START: e.get(ENTITY_ATTRIBUTE_START),
                    EntityKeys.END: e.get(ENTITY_ATTRIBUTE_END)
                }
            )
        entities.append(es)

    return NLUdataset(texts, intents, entities)


def write_data(dataset: NLUdataset, filepath: str):
    """
    Write dataset in Rasa's yml format.

    :param dataset: Dataset to be converted
    :param filepath: Path of the output yml file
    """
    messages = []
    for text, intent, entities in dataset:
        example = {
            TEXT: text,
            INTENT: intent if intent is not None else "default_intent",
            ENTITIES: [
                {
                    ENTITY_ATTRIBUTE_TYPE: e[EntityKeys.TYPE],  # key 'entity'
                    ENTITY_ATTRIBUTE_START: e[EntityKeys.START],  # key 'start'
                    ENTITY_ATTRIBUTE_END: e[EntityKeys.END],  # key 'end'
                    ENTITY_ATTRIBUTE_VALUE: text[e[EntityKeys.START]:e[EntityKeys.END]]  # key 'value'
                }
                for e in entities
            ],
        }
        message = Message(data=example)
        messages.append(message)

    training_data = TrainingData(training_examples=messages)
    mry = rasa_yaml.RasaYAMLWriter()
    md = mry.training_data_to_dict(training_data)
    write_yaml(md, filepath)
