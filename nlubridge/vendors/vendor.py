# Copyright (c) 2021 Yaser Martinez-Palenzuela, Deutsche Telekom AG
# This software is distributed under the terms of the MIT license
# which is available at https://opensource.org/licenses/MIT

from ..nlu_dataset import NluDataset


class Vendor:
    """Abstract class for a vendor."""

    @property
    def name(self):
        """Return the class's name as a hint to the vendor name."""
        return self.__class__.__name__

    @property
    def alias(self):
        """Return alias, which is a customizable name of the instance."""
        return self._alias

    @alias.setter
    def alias(self, value):
        """Set alias, which is a customizable name of the instance."""
        self._alias = value

    def train(self, dataset: NluDataset):
        """
        Train intent and/or entity classification.

        :param dataset: Training data
        """
        raise NotImplementedError

    def test(self, dataset: NluDataset) -> NluDataset:
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
