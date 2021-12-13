# Copyright (c) 2021 Yaser Martinez-Palenzuela, Deutsche Telekom AG
# This software is distributed under the terms of the MIT license
# which is available at https://opensource.org/licenses/MIT

import re
import tempfile
import os
import logging

from fasttext import train_supervised

from .vendors import Vendor

config = {}

logger = logging.getLogger(__name__)
DEFAULT_CONFIG = {
    "wordNgrams": 1,
    "dim": 100,
    "verbose": 2,
    "ws": 5,
    "neg": 10,
    "bucket": 50000,
    "loss": "softmax",
    "minn": 4,
    "maxn": 5,
    "t": 0.01,
    "minCount": 1,
}


class FastText(Vendor):
    alias = "fasttext"

    def __init__(self, epochs=10000, lr=0.1, config=DEFAULT_CONFIG):
        """
        Interface for the FastText classifier.

        See `FastText classifier <https://github.com/facebookresearch/fastText/>`_.

        Features:
            * bag of n-grams as additional features to capture some
              partial information about the local word order
            * fast and memory efficient mapping of the n-grams by using
              the hashing trick
            * Leverages subword information (via character n-gram)
            * Can build representations for out of vocabulary tokens
              (char ngrams average)
            * Fast training with multicore support implementation

        Pros:
            * Competitive performance, specially on low data regime
            * Efficient both at training and at test time
            * Possibility of incorporating domain knowledge
            * Open source

        **Reference:**

        `Paper: <https://arxiv.org/abs/1607.01759>`_

        :param epochs: training epochs
        :type epochs: int
        :param lr: learning rate
        :type lr: float
        :param config: dictionary with additional parameters
        :type config: dict
        """
        self.epochs = epochs
        self.lr = lr
        self.model = None
        self.config = config

    def train_intent(self, dataset):
        """Train intent classifier."""
        train_data = self._convert(dataset)
        logger.info(f"Training on {dataset.n_samples} samples")
        self.model = train_supervised(
            input=train_data, epoch=self.epochs, lr=self.lr, **self.config
        )
        # remove tempfile
        os.remove(train_data)
        return self

    def test_intent(self, dataset, return_probs=False):
        """Test intent classifier."""
        logger.info(f"Testing on {dataset.n_samples} samples")
        intents = []
        probs = []
        for text in dataset.texts:
            text = self._clean_text(text)
            text = "".join(text.splitlines())
            result = self.model.predict(text)
            intent = result[0][0]
            intent = intent[len("__label__") :]
            prob = result[1][0]
            intents.append(intent)
            probs.append(prob)
        if return_probs:
            return intents, probs
        return intents

    def _convert(self, dataset):
        lines = []
        for text, intent, _ in dataset:
            text = self._clean_text(text)
            line = f"__label__{intent} {text}"
            lines.append(line)
        lines = "\n".join(lines)

        with tempfile.NamedTemporaryFile(mode="w", encoding="utf8", delete=False) as f:
            f.write(lines)
            train_data = f.name

        return train_data

    @staticmethod
    def _clean_text(text):
        return re.sub("[^a-zA-Z\n.]", " ", text)
