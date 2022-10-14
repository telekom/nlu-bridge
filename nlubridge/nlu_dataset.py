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
    INTENT = "intent"
    CONFIDENCE = "confidence"


class NluDataset:
    def __init__(
        self,
        texts: List[str],
        intents: Optional[List[str]] = None,
        entities: Optional[
            List[List[Dict]]
        ] = None,  # TODO: Better code entities as custom objects?
        n_best_intents: Optional[
            List[List[Dict]]
        ] = None,  # TODO: Better code n-best-list items as custom objects?
        max_intent_length=None,
    ) -> None:
        """
        Class for managing NLU data with intent and entity annotations.

        :param texts: A list of utterances
        :type texts: List[str]
        :param intents: A list of intents for each text. If intents is not
            empty then there needs to be an intent for each text.
        :type intents: List[str]
        :param entities: If intents is not empty then there needs to be an
            intent for each text.
        :type entities: List[dict]
        :param n_best_intents: List of dicts with intents and prediction
        :param max_intent_length: If given the intents will be strimmed to
            the first `max_intent_length` in chars. This is required
            because some vendors like LUIS don't accept very long intent names.
        :type max_intent_length: int or None
        """
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

    # TODO: arguments should not have these defaults
    def sample(
        self,
        size: Union[float, int],
        random_state: Optional[int] = None,
        stratification: Any = "intents",
    ) -> NluDataset:
        """
        Sample the dataset preserving intent proportions.

        :param size: can be an integer representing the final number of
            samples or a float representing the ratio
        :type size: float or int
        :param random_state: seed for random processes
        :type random_state: int
        :param stratification: If not None, data is split in a stratified fashion, using
            this as the class labels. Default is using the intent labels for
            stratification.
        """
        if isinstance(size, numbers.Integral):
            size = size / self.n_samples

        _, sampled = self.train_test_split(
            test_size=size, random_state=random_state, stratification=stratification
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
        :type excluded: List[str]
        :param allowed: List of intents for which to return data
        :type allowed: List[str]
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
        :type n: int
        """
        allowed = self.unique_intents[:n]
        ds = self.filter_by_intent_name(allowed=allowed)
        return ds

    # TODO: Fix docstring
    def clip_by_intent_frequency(
        self, max_frequency: int, min_frequency: int = None
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
        :type max_frequency: int
        :param min_frequency: maximum number of samples per intent to
            include in the returned dataset. If an intent has less
            samples, it will be dropped.
        :type min_frequency: int

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
        shuffle: bool = False,  # This is not required. Can use shuffle() to shuffle
    ) -> NluDataset:
        """
        Return a smaller dataset with similar intent distribution.

        Only a certain percentage of each intent's data is kept, while
        for intents with fewer examples that percentage is higher.

        Intents with less than min_frequency samples will not be
        reduced. For all others, the target count is computed as
        `min_freq + (num_samples - min_freq) * target_rate`
        """
        if shuffle:
            raise NotImplementedError(
                "shuffle=True has not yet been implemented. You can "
                "use NLUdataset.shuffle() to shuffle the dataset "
                "before subsampling."
            )
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

    # TODO: Can we get rid of arguments test_size and random_state because they are
    #     "inherited" from sklearn method of same name and can be used with **args?
    def train_test_split(
        self, test_size=None, random_state=None, stratification="intents", **args
    ) -> Tuple[NluDataset, NluDataset]:
        """
        Split dataset into train and test partitions.

        :param test_size: fraction of samples to use for testing
        :type test_size: float
        :param random_state: random seed
        :type random_state: int
        :param stratification: TODO
        :type stratification: TODO
        :param args: additional args for sklearn's train_test_split
            provided as dictionary
        :type args: dict
        """

        def configure_stratification():
            """
            Allow configuring stratification options.

            The default setting, "intents", is used when we want to
            use stratification by intent strings. If we want to use
            anything else, we can specify it in the (parent) method's
            stratification parameter, e.g. None, which would turn
            stratification off (using the underlying
            sklearn.model_selection.train_test_split() method).

            The default setting, which is passed as a string, will be
            converted into the appropriate self.intents, which isn't
            accessible from the parent method's parameters.

            Returns the stratify setting (either self.intents or
            whatever is passed) to the sklearn train_test_split
            method.
            """
            if stratification == "intents":
                stratify = self.intents
            else:
                stratify = stratification
            return stratify

        (data_train, data_test) = train_test_split(
            self._data,
            stratify=configure_stratification(),
            test_size=test_size,
            random_state=random_state,
            **args,
        )
        train_ds = NluDataset._from_data(data_train)
        test_ds = NluDataset._from_data(data_test)
        return train_ds, test_ds

    def cross_validation_splits(
        self, cv_iterator: BaseCrossValidator = None
    ) -> Iterator[Tuple[NluDataset, NluDataset]]:
        """Cross validation generator function."""
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

    def to_json(self, path: Union[Path, str] = None) -> Optional[List[Dict]]:
        """
        Convert dataset to JSON.

        If a path is given, saves to that file. Otherwise returns records as a list of
        dicts.

        :param path: optional path under which to save the JSON file
        :type path: Union[Path, str]
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
