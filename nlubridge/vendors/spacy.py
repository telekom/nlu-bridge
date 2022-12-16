# Copyright (c) 2021 Yaser Martinez-Palenzuela, Deutsche Telekom AG
# This software is distributed under the terms of the MIT license
# which is available at https://opensource.org/licenses/MIT

import logging
import random

import spacy
from spacy.pipeline.textcat import DEFAULT_SINGLE_TEXTCAT_MODEL
from spacy.training import Example

from .vendor import Vendor


logger = logging.getLogger(__name__)


class Spacy(Vendor):
    def __init__(self, n_iter=10, config=None, language="en"):
        """
        Interface for the Spacy intent classifier.

        :param n_iter: Number of training iterations (default: 1000)
        :type n_iter: int
        :param config: Textcat config in dictionary format as described
            at https://spacy.io/api/textcategorizer
        :type config: dict
        :param language: Language string (default: "en")
        :type language: str
        """
        self._alias = self.name
        self._nlp = spacy.blank(language)
        if not config:
            config = {
                "threshold": 0.5,
                "model": DEFAULT_SINGLE_TEXTCAT_MODEL,
            }
        self._textcat = self._nlp.add_pipe("textcat", config=config)
        self._n_iter = n_iter

    def _convert(self, dataset):
        examples = []
        for text, intent in zip(dataset.texts, dataset.intents):
            doc = self._nlp.make_doc(text)
            cats = {
                intent: True if intent == each else False
                for each in dataset.unique_intents
            }
            example = Example.from_dict(doc, {"cats": cats})
            examples.append(example)
        return examples

    def train_intent(self, dataset):
        """Train intent classifier."""
        examples = self._convert(dataset)

        get_examples = lambda: examples  # noqa: E731
        optimizer = self._nlp.initialize(get_examples)
        for itn in range(self._n_iter):
            random.shuffle(examples)
            for example in examples:
                self._nlp.update([example], sgd=optimizer)

    def test_intent(self, dataset, return_probs=False):
        """Test intent classifier."""
        intents = []
        probs = []
        for text in dataset.texts:
            doc = self._nlp(text)
            intent = max(doc.cats, key=doc.cats.get)
            prob = doc.cats[intent]
            intents.append(intent)
            probs.append(prob)
        if return_probs:
            return intents, probs
        return intents
