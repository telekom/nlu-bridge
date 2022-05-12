# Copyright (c) 2021 Yaser Martinez-Palenzuela, Deutsche Telekom AG
# This software is distributed under the terms of the MIT license
# which is available at https://opensource.org/licenses/MIT

import random

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score, classification_report, f1_score

from ..datasets import OUT_OF_SCOPE_TOKEN, NLUdataset


class Vendor(BaseEstimator, ClassifierMixin):
    """Abstract class for a vendor."""

    # TODO: Is this really a reliable method for the purpose?
    def fake_train_intent(self, data):
        """
        Train vendor with noise.

        This method has the purpose of resetting any cache that
        a vendor system might be using to memorize labels
        """
        texts = data.texts
        intents = data.intents
        random.shuffle(intents)
        fake_data = NLUdataset(texts, intents, entities=[])
        self.train_intent(fake_data)

    def train(self, dataset: NLUdataset):
        """
        Train intent and/or entity classification.

        :param dataset: Training data
        """
        raise NotImplementedError

    def test(self, dataset: NLUdataset) -> NLUdataset:
        """
        Test a given dataset and obtain classification results as NLUdataset.

        The returned NLUdataset will include intent and/or entity predictions, depending
        on what the model can handle and has been trained on.

        :param dataset: Input dataset to be tested
        :return: NLUdataset object comprising the classification results
        """
        raise NotImplementedError

    def train_intent(self, dataset):
        """Train intent classification."""
        raise NotImplementedError

    def test_intent(self, dataset):
        """Test intent classification."""
        raise NotImplementedError

    @property
    def name(self):
        """Return the class's name as a hint to the vendor name."""
        return self.__class__.__name__

    def f1_score(self, test_dataset, average="micro"):
        """
        Predict intent f1 score.

        Predict intents for samples in test_dataset and compute
        f1 score by comparing predictions to labels given in dataset
        """
        y_pred = self.test_intent(test_dataset)
        y_true = test_dataset.intents
        return f1_score(y_true, y_pred, average=average)

    def out_of_scope_accuracy(self, test_dataset):
        """Test OOS accuracy."""
        y_pred = self.test_intent(test_dataset)
        y_true = np.full_like(y_pred, OUT_OF_SCOPE_TOKEN)
        return accuracy_score(y_true, y_pred)

    def classification_report(self, test_dataset):
        """
        Compute intent classification report.

        Predict intents for samples in test_dataset and compute a
        scikit-learn classification report by comparing predictions to
        labels given in dataset.
        """
        y_pred = self.test_intent(test_dataset)
        y_true = test_dataset.intents
        return classification_report(y_true, y_pred)

    def fit(self, X, y=None):
        """
        scikit-learn compatibility.

        This method directly accepts texts and intents instead of a
        NLUdataset.
        """
        texts = X
        intents = y
        ds = NLUdataset(texts, intents)
        self.train_intent(ds)
        return self

    def predict(self, X, y=None):
        """
        scikit-learn compatibility.

        This method directly accepts texts and intents instead of a
        dataset.
        """
        texts = X
        intents = y
        ds = NLUdataset(texts, intents)
        y = self.test_intent(ds)
        return y

    def score(self, X, y=None, average="micro"):
        """scikit-learn compatibility."""
        texts = X
        intents = y
        ds = NLUdataset(texts, intents)
        return self.f1_score(ds, average=average)
