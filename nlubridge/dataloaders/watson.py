# Copyright (c) 2021 Klaus-Peter Engelbrecht, Deutsche Telekom AG
# This software is distributed under the terms of the MIT license
# which is available at https://opensource.org/licenses/MIT

from nlubridge.datasets import NluDataset

from .utils import from_csv


def from_watson(filepath) -> NluDataset:
    """Load data from Watson format as NLUdataset."""
    return from_csv(filepath, text_col=0, intent_col=1)
