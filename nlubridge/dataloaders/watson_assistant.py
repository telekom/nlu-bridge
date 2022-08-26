# Copyright (c) 2021 Klaus-Peter Engelbrecht, Deutsche Telekom AG
# This software is distributed under the terms of the MIT license
# which is available at https://opensource.org/licenses/MIT

from nlubridge.nlu_dataset import NluDataset

from .utils import from_csv


def from_watson_assistant(filepath) -> NluDataset:
    """Load data from Watson format as NLUdataset."""
    return from_csv(filepath, text_col=0, intent_col=1)
