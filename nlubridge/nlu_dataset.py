# Copyright (c) 2021 Yaser Martinez-Palenzuela, Deutsche Telekom AG
# This software is distributed under the terms of the MIT license
# which is available at https://opensource.org/licenses/MIT

from __future__ import annotations

import collections
import itertools
import json
import numbers
import random
import warnings
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

import numpy as np
from sklearn.model_selection import (
    BaseCrossValidator,
    StratifiedKFold,
    train_test_split,
)


OUT_OF_SCOPE_TOKEN = "out_of_scope"


class EntityKeys:
    """Keys for entity dicts in NLUdataset."""

    TYPE = "entity"
    START = "start"
    END = "end"
    CONFIDENCE = "confidence"


class NBestKeys:
    """Keys for n-best list items (dicts) in NLUdataset."""

    INTENT = "intent"
    CONFIDENCE = "confidence"


class NluDataset:
    def __init__(
        self,
        texts: List[str],
        intents: Optional[List[str]] = None,
        entities: Optional[List[List[Dict]]] = None,
        n_best_intents: Optional[List[List[Dict]]] = None,
        max_intent_length=None,
    ) -> None:
        """
        Class for managing NLU data with intent and entity annotations.

        :param texts: A list of utterances
        :param intents: A list of intents, one for each text. If intents is
            not None then there needs to be an intent for each text.
        :param entities: List of entities for each text. If not None then
            there needs to be a List for each text.
        :param n_best_intents: List of n-best results (each denoted as a dict
            having intent and confidence keys) for each text. If not None then
            there needs to be a List for each text.
        :param max_intent_length: If given the intents will be trimmed to
            the first `max_intent_length` chars. This is required
            because some vendors like LUIS don't accept very long intent names.
        """
        # Cast to list so we can be a bit more flexible with the inputs, e.g. use a
        # Pandas Series
        texts = list(texts) if texts is not None else []
        intents = list(intents) if intents is not None else None
        entities = list(entities) if entities is not None else None

        ds_intents = (
            self._prepare_intents(max_intent_length, intents)
            if intents
            else [None for _ in texts]
        )
        ds_entities = entities if entities else [list() for _ in texts]
        ds_n_best_intents = (
            n_best_intents if n_best_intents else [list() for _ in texts]
        )
        self._data = list(zip(texts, ds_intents, ds_entities, ds_n_best_intents))
        self._unique_intents = self._get_unique_intents()
        self._intent_frequencies: collections.Counter = collections.Counter(
            self.intents
        )
        self._unique_entities = self._get_unique_entities()

    @property
    def name(self) -> str:
        """Return the class name."""
        return self.__class__.__name__

    @property
    def unique_intents(self) -> List[str]:  # noqa: D102
        return self._unique_intents

    @property
    def intent_frequencies(self) -> collections.Counter:  # noqa: D102
        return self._intent_frequencies

    @property
    def unique_entities(self) -> List[str]:  # noqa: D102
        return self._unique_entities

    @property
    def texts(self) -> List[str]:  # noqa: D102
        return [record[0] for record in self._data]

    @property
    def intents(self) -> List[str]:  # noqa: D102
        return [record[1] for record in self._data]

    @property
    def entities(self) -> List[List[Dict]]:  # noqa: D102
        return [record[2] for record in self._data]

    @property
    def confidences(self) -> List[Optional[float]]:  # noqa: D102
        try:
            confs = [record[3][0].get(NBestKeys.CONFIDENCE) for record in self._data]
        except IndexError:
            confs = [None for _ in self._data]
        return confs

    @property
    def n_best_intents(self) -> List[List[Dict]]:  # noqa: D102
        return [record[3] for record in self._data]

    @property
    def n_samples(self) -> int:  # noqa: D102
        return len(self._data)

    @property
    def n_intents(self) -> int:  # noqa: D102
        return len(self.unique_intents)

    @property
    def n_entities(self) -> int:  # noqa: D102
        return len(self.unique_entities)

    @classmethod
    def from_joined(cls, *datasets: NluDataset) -> NluDataset:
        """Join NLUdatasets provided in a list."""
        warnings.warn(
            "This method will be removed in a future version. Please use "
            "nlubridge.concat() or NluDataset.join().",
            DeprecationWarning,
        )
        data = []
        for dataset in datasets:
            data.extend(dataset._data)
        joint_dataset = NluDataset._from_data(data)
        return joint_dataset

    @classmethod
    def _from_data(cls, data):
        return cls(*zip(*data))

    def _get_unique_intents(self):
        # unique_intents = list(dict.fromkeys(intents))  # old
        unique_intents = sorted(
            [intent for intent in list(set(self.intents)) if intent is not None]
        )
        return unique_intents

    def _get_unique_entities(self):
        it = itertools.chain.from_iterable(self.entities)
        entity_types = [
            entity[EntityKeys.TYPE]
            for entity in it
            if isinstance(entity, dict) and EntityKeys.TYPE in entity
        ]
        return sorted(list(set(entity_types)))

    @staticmethod
    def _prepare_intents(max_intent_length, intents):
        if not max_intent_length:
            # Return a copy of intents to avoid interference with usages outside this
            # class; also makes sure we have a list
            return list(intents)
        cropped_intents = [intent[:max_intent_length] for intent in intents]
        if len(set(intents)) > len(set(cropped_intents)):
            warnings.warn(
                "Intent names are not unique on first {} characters".format(
                    max_intent_length
                )
            )
        return cropped_intents

    def __iter__(self):
        return iter(self._data)

    def __getitem__(self, key):
        if isinstance(key, slice):
            data = self._data[key]
            return NluDataset._from_data(data)

        elif isinstance(key, collections.abc.Sequence) or isinstance(key, np.ndarray):
            data = [self._data[each] for each in key]
            return NluDataset._from_data(data)

        elif isinstance(key, numbers.Integral):
            data = [self._data[key]]
            return NluDataset._from_data(data)

    def __len__(self):
        return len(self._data)

    def __add__(self, other):
        return NluDataset._from_data(self._data + other._data)

    def join(self, other: NluDataset) -> NluDataset:
        """Join this NluDataset with another one."""
        return self + other

    def shuffle(self) -> NluDataset:
        """Shuffles the dataset and returns itself."""
        random.shuffle(self._data)
        return self

    def sample(
        self,
        size: Union[float, int],
        random_state: Optional[int] = None,
        stratify: Any = "intents",
    ) -> NluDataset:
        """
        Sample the dataset preserving intent proportions.

        :param size: can be an integer representing the final number of
            samples or a float representing the ratio
        :param random_state: seed for random processes
        :param stratify: If not None, data is split in a stratified fashion, using
            this as the class labels. Default is using the intent labels for
            stratification.
        """
        if isinstance(size, numbers.Integral):
            size = size / self.n_samples

        _, sampled = self.train_test_split(
            test_size=size, random_state=random_state, stratify=stratify
        )
        return sampled

    def filter_by_intent_name(
        self,
        excluded: Optional[List[str]] = None,
        allowed: Optional[List[str]] = None,
    ) -> NluDataset:
        """
        Filter the dataset by intents.

        :param excluded: List of intents for which not to return data
        :param allowed: List of intents for which to return data
        """
        if excluded is None:
            excluded = []
        if allowed is None:
            allowed = self.unique_intents

        data = []
        for intent, record in zip(self.intents, self._data):
            if intent in excluded:
                continue
            if intent not in allowed:
                continue
            data.append(record)
        return NluDataset._from_data(data)

    def select_first_n_intents(self, n: int) -> NluDataset:
        """
        Return a dataset with a max of `n` intents from the dataset.

        The order of the selected intents is given by
        `self.unique_intents`

        :param n: Number of intents to return
        """
        allowed = self.unique_intents[:n]
        ds = self.filter_by_intent_name(allowed=allowed)
        return ds

    def clip_by_intent_frequency(
        self, max_frequency: int, min_frequency: Optional[int] = None
    ) -> NluDataset:
        """
        Sample the dataset leaving only `max_freq` samples per intent.

        If min_frequency is given then intents with less than that
        number of samples will be dropped.

        Example:
        -------
        To get a dataset with all intents having the same number of
        samples you can do:

        >>> ds = MyNluDataset()
        >>> ds.intent_frequencies
        Counter({'telefonie_disambiguierung': 562,
                'telefonie_beidseitig': 869,
                'internet_kein_internet': 815,
                'internet_langsam': 764,
                'internet_abbrueche': 654,
                'komplettausfall_eindeutig': 931})

        >>> ds.clip_by_intent_frequency(700, 700).intent_frequencies
        Counter({'telefonie_beidseitig': 700,
                'internet_kein_internet': 700,
                'internet_langsam': 700,
                'komplettausfall_eindeutig': 700})


        :param max_frequency: maximum number of samples per intent to
            include in the returned dataset
        :param min_frequency: minimum number of samples per intent to
            include in the returned dataset. If an intent has less
            samples, it will be dropped.

        """
        freqs: collections.Counter = collections.Counter()
        data = []
        for intent, record in zip(self.intents, self._data):
            if (
                min_frequency is not None
                and self.intent_frequencies[intent] <= min_frequency
            ):
                continue

            if freqs[intent] == max_frequency:
                continue

            freqs[intent] += 1
            data.append(record)
        return NluDataset._from_data(data)

    def subsample_by_intent_frequency(
        self,
        target_rate: float,
        min_frequency: int,
    ) -> NluDataset:
        """
        Return a smaller dataset with similar intent distribution.

        Only a certain percentage of each intent's data is kept, while
        for intents with fewer examples that percentage is higher.

        Intents with less than min_frequency samples will not be
        reduced. For all others, the target count is computed as
        `min_freq + (num_samples - min_freq) * target_rate`

        :param target_rate: fraction of data to keep beyond min_frequency
        :param min_frequency: number of utterance that are always kept
        """
        freqs: collections.Counter = collections.Counter()
        data = []
        for intent, record in zip(self.intents, self._data):
            intent_freq = self.intent_frequencies[intent]
            target_freq = min_frequency + ((intent_freq - min_frequency) * target_rate)
            is_below_min = intent_freq <= min_frequency
            is_below_target = freqs[intent] < target_freq
            if is_below_min or is_below_target:
                freqs[intent] += 1
                data.append(record)
        return NluDataset._from_data(data)

    def train_test_split(
        self, stratify="intents", **args
    ) -> Tuple[NluDataset, NluDataset]:
        """
        Split dataset into train and test partitions.

        Uses sklearn.model_selection.train_test_split under the hood. Arguments are
        directly passed to the sklearn function, with the exception of `stratify`: this
        is set by default to "intents" which will set stratify to self.intents (i.e.,
        stratify by intents). Otherwise, stratify works like with sklearn.

        :param stratify: like sklearn train_test_split argument `stratify`.
            Defaults to using self.intents.
        :param args: additional arguments for sklearn's train_test_split, see
            https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
        """
        if stratify == "intents":
            stratify = self.intents
        (data_train, data_test) = train_test_split(
            self._data,
            stratify=stratify,
            **args,
        )
        train_ds = NluDataset._from_data(data_train)
        test_ds = NluDataset._from_data(data_test)
        return train_ds, test_ds

    def cross_validation_splits(
        self, cv_iterator: BaseCrossValidator = None
    ) -> Iterator[Tuple[NluDataset, NluDataset]]:
        """
        Cross validation generator function.

        :param cv_iterator: sklearn cross validator object to control cv folds
        """
        if cv_iterator is None:
            cv_iterator = StratifiedKFold(n_splits=5, shuffle=True)
        for train_idx, test_idx in cv_iterator.split(self.texts, self.intents):
            train_ds = self[train_idx]
            test_ds = self[test_idx]
            yield train_ds, test_ds

    def to_records(self):
        """Deprectaed. Use to_json() instead."""
        warnings.warn(
            "This method will be removed in a future version. Please use "
            "NluDataset.to_json().",
            DeprecationWarning,
        )
        return self.to_json()

    def to_dict(self) -> Dict[str, list]:
        """
        Return intent data as dict.

        Intents are keys, each holding a list of corresponding
        utterances.
        """
        d = collections.defaultdict(list)
        for record in self._data:
            d[record[1]].append(record[0])
        return dict(d)

    def to_json(self, path: Optional[Union[Path, str]] = None) -> Optional[List[Dict]]:
        """
        Convert dataset to JSON.

        If a path is given, saves to that file. Otherwise returns records as a list of
        dicts.

        :param path: optional path under which to save the JSON file
        """
        records = [
            {"text": text, "intent": intent, "entities": entities, "n_best_list": nbest}
            for text, intent, entities, nbest in self._data
        ]

        if path is None:
            return records

        with open(path, "w") as f:
            json.dump(records, f, indent=4, ensure_ascii=False)
        return None


def concat(dataset_list: List[NluDataset]) -> NluDataset:
    """
    Concatenate NluDatasets.

    :param dataset_list: list of NluDataset objects
    """
    joined = NluDataset([])
    for ds in dataset_list:
        joined += ds
    return joined
