# Copyright (c) 2022 Ralf Kirchherr, Deutsche Telekom AG
# This software is distributed under the terms of the MIT license
# which is available at https://opensource.org/licenses/MIT
from __future__ import annotations

import logging
import os
import pathlib
import tempfile
from typing import List, Optional, Tuple, Union

from rasa.core.agent import Agent
from rasa.core.channels.channel import UserMessage
from rasa.model import get_local_model
from rasa.model_training import train_nlu
from rasa.shared.nlu.constants import (
    INTENT,
    INTENT_NAME_KEY,
    INTENT_RANKING_KEY,
    PREDICTED_CONFIDENCE_KEY,
)

from nlubridge import NBestKeys, NluDataset, to_rasa
from nlubridge.dataloaders.rasa import convert_entities_to_nludataset
from nlubridge.vendors.vendor import Vendor


logger = logging.getLogger(__name__)

DEFAULT_INTENT_RASA_CONFIG_PATH = os.path.join(
    pathlib.Path(__file__).parent.absolute(), "config", "rasa_nlu_config.yml"
)


class Rasa3(Vendor):
    def __init__(self, model_config: Optional[str] = None):
        """
        Interface for the `Rasa NLU <https://github.com/RasaHQ/rasa>`_.

        Uses the default pipeline as of January 22 2021. No algorithmic
        details have been researched. The configuration file can be
        found in the config directory within this directory. A custom
        pipeline can be provided as argument to the class constructor.

        :param model_config: filepath to a Rasa config file
        """
        self._alias = self.name
        self._config = model_config
        self._agent = None

    @classmethod
    def from_model(cls, model_path: Union[str, pathlib.Path]) -> Rasa3:
        """
        Construct a Rasa3 object from a saved model.

        Loading a model can fail if versions of dependencies differ between the
        environment where the model was trained and the environment where the model
        is loaded.

        :param model_path: file path to a saved rasa model.
        :return: Rasa
        """
        obj = cls()
        # Since we don't know the config used to train the model, make sure we don't
        # add an inconsistent config accidentally should we ever change __init__().
        assert obj._config is None
        model_archive = get_local_model(model_path)
        obj._agent = Agent.load(model_path=model_archive)
        return obj

    def train(self, dataset: NluDataset) -> Rasa3:
        """
        Train intent and/or entity classification.

        :param dataset: Training data
        :return: It's own Rasa3 object
        """
        self._config = self._config if self._config else DEFAULT_INTENT_RASA_CONFIG_PATH
        with tempfile.TemporaryDirectory() as tmpdirname:
            nlu_yml_file = os.path.join(
                pathlib.Path(tmpdirname), "nlu.yml"
            )  # output path for temporary nlu.yml
            to_rasa(dataset, nlu_yml_file)
            logger.info(f"Start training (using {self._config!r})...")
            model_archive = train_nlu(self._config, nlu_yml_file, tmpdirname)
            logger.info("Training completed!")

            logger.info("Load model...")
            self._agent = Agent.load(model_path=model_archive)
            logger.info("Model loaded!")
        return self

    def train_intent(self, dataset: NluDataset) -> Rasa3:
        """
        Train intent classification.

        This method is mainly for compatibility reasons, as it in case of Rasa identical
        to the `train` method.

        :param dataset: Training data
        :return: It's own Rasa3 object
        """
        return self.train(dataset)

    def _parse_message(self, text):
        # result = asyncio.run(
        #     self._agent.parse_message(text)
        # )  # agent's parse method is a coroutine

        # To avoid errors in Jupyter if  an event loop is already running, don't use the
        # async method from self._agent
        message = UserMessage(text)
        parse_data = self._agent.processor._parse_message_with_graph(message)
        self._agent.processor._check_for_unseen_features(parse_data)
        result = parse_data
        return result

    def test(self, dataset: NluDataset) -> NluDataset:
        """
        Test a given dataset.

        Test a given dataset and obtain the intent and/or entity classification results
        in the NLUdataset format

        :param dataset: Input dataset to be tested
        :return: NLUdataset object comprising the classification results. The list of
            the predicted intent classification probabilities are accessible via the
            additional attribute 'probs' (List[float]).
        """
        if self._agent is None:
            raise RuntimeError("Rasa3 classifier has to be trained first!")
        intents = []
        n_best_lists = []
        entities_list = []
        for text in dataset.texts:
            result = self._parse_message(text)
            intent = result.get(INTENT, {}).get(INTENT_NAME_KEY)
            entities = convert_entities_to_nludataset(result)
            nbest = [
                {
                    NBestKeys.INTENT: ranked.get(INTENT_NAME_KEY),
                    NBestKeys.CONFIDENCE: ranked.get(PREDICTED_CONFIDENCE_KEY),
                }
                for ranked in result.get(INTENT_RANKING_KEY, [])
            ]

            intents.append(intent)
            n_best_lists.append(nbest)
            entities_list.append(entities)

        res = NluDataset(dataset.texts, intents, entities_list, n_best_lists)
        return res

    def test_intent(
        self, dataset: NluDataset, return_probs: bool = False
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
        if self._agent is None:
            raise RuntimeError("Rasa3 classifier has to be trained first!")
        intents = []
        probs = []
        for text in dataset.texts:
            result = self._parse_message(text)
            intent = result.get(INTENT, {}).get(INTENT_NAME_KEY)
            prob = result.get(INTENT, {}).get(PREDICTED_CONFIDENCE_KEY)
            intents.append(intent)
            probs.append(prob)
        if return_probs:
            res = (intents, probs)
        else:
            res = intents
        return res
