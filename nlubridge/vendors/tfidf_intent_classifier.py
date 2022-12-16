# Copyright (c) 2021 Yaser Martinez-Palenzuela, Deutsche Telekom AG
# This software is distributed under the terms of the MIT license
# which is available at https://opensource.org/licenses/MIT

import logging

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline

from .vendor import Vendor


logger = logging.getLogger(__name__)


class TfidfIntentClassifier(Vendor):
    def __init__(self):
        """
        Interface to a TFIDF text classifier.

        Simple word based tf-idf plus a SVM classifier. It can be used
        as a baseline and to quickly tests a script by providing a fast
        algorithm.

        Pros:
            * Very fast classifier
            * Works relatively well for most problems
            * Provides a good baseline that custom models should beat

        Cons:
            * Can't use external information (ex: pretrained word
              embeddings)
        """
        self._alias = self.name
        self._clf = Pipeline(
            [
                ("vect", CountVectorizer()),
                ("tfidf", TfidfTransformer()),
                (
                    "clf",
                    SGDClassifier(
                        loss="log",
                        penalty="l2",
                        alpha=1e-5,
                        random_state=42,
                        max_iter=700,
                        tol=None,
                        n_jobs=-1,
                    ),
                ),
            ]
        )

    def train_intent(self, dataset):
        """Train intent classifier."""
        logger.info(f"Training on {dataset.n_samples} samples")
        X = dataset.texts
        y = dataset.intents
        self._clf.fit(X, y)
        return self

    def test_intent(self, dataset, return_probs=False):
        """Test intent classifier."""
        logger.info(f"Testing on {dataset.n_samples} samples")
        X = dataset.texts
        probs = self._clf.predict_proba(X)
        pred_idxs = probs.argmax(axis=1)
        winner_class_probs = probs.max(axis=1)
        winner_class_probs = list(winner_class_probs)
        intents = self._clf.classes_[pred_idxs]
        intents = list(intents)

        if return_probs:
            return intents, winner_class_probs
        return intents
