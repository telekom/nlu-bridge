# Copyright (c) 2021 Yaser Martinez-Palenzuela, Deutsche Telekom AG
# This software is distributed under the terms of the MIT license
# which is available at https://opensource.org/licenses/MIT

import collections
import itertools
import json
import numbers
import random
from pathlib import Path
from typing import List
import warnings

from sklearn.model_selection import train_test_split, StratifiedKFold
import numpy as np


DATA_PATH = Path(__file__).parents[2] / "data"
OUT_OF_SCOPE_TOKEN = "out_of_scope"


class EntityKeys:
    """Keys for entity dicts in NLUdataset."""

    TYPE = "entity"
    START = "start"
    END = "end"


class NLUdataset:
    def __init__(
        self,
        texts: List[str],
        intents: List[str] = None,
        entities: List[List[dict]] = None,
        out_of_scope=False,
        max_intent_length=None,
        seed=42,
    ):
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
        :param max_intent_length: If given the intents will be strimmed to
            the first `max_intent_length` in chars. This is required
            because some vendors like LUIS don't accept very long intent names.
        :type max_intent_length: int or None
        :param seed: The initial random seed used by shuffle.
        :type seed: int
        """
        texts = list(texts) if texts is not None else []
        intents = list(intents) if intents is not None else None
        entities = list(entities) if entities is not None else None

        self.texts = texts
        self.n_samples = len(self.texts)

        if not intents or intents is None:
            self.unique_intents = None  # TODO: Can this be empty list?
            if out_of_scope:
                self.unique_intents = [OUT_OF_SCOPE_TOKEN]
            self.n_intents = None  # TODO: Can this be 0?
            self.intents = [None for _ in texts]
        else:
            if max_intent_length:
                self.intents = [intent[:max_intent_length] for intent in intents]
                if len(set(intents)) > len(set(self.intents)):
                    warnings.warn(
                        "Intent names are not unique on first {} characters".format(
                            max_intent_length
                        )
                    )
            else:
                self.intents = intents
            self.unique_intents = list(dict.fromkeys(self.intents))

            if out_of_scope:
                self.unique_intents += [OUT_OF_SCOPE_TOKEN]

            self.n_intents = len(self.unique_intents)
            self.intent_frequencies = collections.Counter(self.intents)

        self.entities = entities
        if not entities:
            self.unique_entities = []
            self.n_entities = 0
            self.entities = [list() for _ in texts]

        if entities is not None:
            it = itertools.chain.from_iterable(entities)
            unique_entities = []
            for each in it:
                if isinstance(each, dict) and EntityKeys.TYPE in each:
                    unique_entities.append(each[EntityKeys.TYPE])
            self.unique_entities = list(dict.fromkeys(unique_entities))
            self.n_entities = len(self.unique_entities)

        self.data = [
            (text, intent, entities)
            for text, intent, entities in zip(self.texts, self.intents, self.entities)
        ]
        # TODO: Can we handle random seed outside this class to avoid confusions
        #  when making implicit copies like in sample() method?
        random.seed(seed)

    def __iter__(self):
        return iter(self.data)

    def __getitem__(self, key):
        if isinstance(key, slice):
            texts = self.texts[key]
            intents = self.intents[key]
            entities = self.entities[key]
            return NLUdataset(texts, intents, entities)

        elif isinstance(key, collections.abc.Sequence) or isinstance(key, np.ndarray):
            texts = [self.texts[each] for each in key]
            intents = [self.intents[each] for each in key]
            entities = [self.entities[each] for each in key]
            return NLUdataset(texts, intents, entities)

        elif isinstance(key, numbers.Integral):
            return self.data[key]

    def __len__(self):
        return len(self.texts)

    def __add__(self, other):
        return NLUdataset(
            texts=self.texts + other.texts,
            intents=self.intents + other.intents,
            entities=self.entities + other.entities,
        )

    @classmethod
    def from_joined(cls, *datasets):
        """Join NLUdatasets provided in a list."""
        texts, intents, entities = [], [], []

        for dataset in datasets:
            texts.extend(dataset.texts)
            intents.extend(dataset.intents)
            entities.extend(dataset.entities)

        joint_dataset = cls(texts, intents, entities)
        return joint_dataset

    @property
    def name(self):
        """Return the class name."""
        return self.__class__.__name__

    def shuffle(self):
        """Shuffles the dataset."""
        random.shuffle(self.data)
        self.texts, self.intents, self.entities = zip(*self.data)
        return self

    def sample(self, size=0.1, random_state=0, stratification="intents"):
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
        self, excluded: List[str] = None, allowed: List[str] = None
    ):
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

        texts, intents = [], []
        for text, intent, _ in self.data:
            if intent in excluded:
                continue

            if intent not in allowed:
                continue

            texts.append(text)
            intents.append(intent)

        return NLUdataset(texts, intents, None)

    def select_first_n_intents(self, n: int):
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

    def clip_by_intent_frequency(self, max_frequency, min_frequency=None):
        """
        Sample the dataset leaving only `max_freq` samples per intent.

        If min_frequency is given then intents with less than that
        number of samples will be dropped.

        Example:
        -------
        To get a dataset with all intents having the same number of
        samples you can do:

        >>> from nlutests.datasets import TDGDataset
        >>> ds = TDGDataset()
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
        freqs = collections.Counter()
        texts, intents = [], []

        for text, intent, _ in self.data:
            if (
                min_frequency is not None
                and self.intent_frequencies[intent] <= min_frequency
            ):
                continue

            if freqs[intent] == max_frequency:
                continue

            freqs[intent] += 1
            texts.append(text)
            intents.append(intent)

        return NLUdataset(texts, intents, None)

    def subsample_by_intent_frequency(self, target_rate, min_frequency, shuffle=False):
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
        freqs = collections.Counter()
        texts, intents, entities = [], [], []
        for text, intent, entity_list in self.data:
            intent_freq = self.intent_frequencies[intent]
            target_freq = min_frequency + ((intent_freq - min_frequency) * target_rate)
            is_below_min = intent_freq <= min_frequency
            is_below_target = freqs[intent] < target_freq
            if is_below_min or is_below_target:
                freqs[intent] += 1
                texts.append(text)
                intents.append(intent)
                entities.append(entity_list)
        return NLUdataset(texts, intents, entities)

    def train_test_split(
        self, test_size=0.25, random_state=0, stratification="intents", **args
    ):
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
            accessible from the parent methodd's parameters.

            Returns the stratify setting (either self.intents or
            whatever is passed) to the sklearn train_test_split
            method.
            """
            if stratification == "intents":
                stratify = self.intents
            else:
                stratify = stratification
            return stratify

        (
            texts_train,
            texts_test,
            intents_train,
            intents_test,
            entities_train,
            entities_test,
        ) = train_test_split(
            self.texts,
            self.intents,
            self.entities,
            stratify=configure_stratification(),
            test_size=test_size,
            random_state=random_state,
            **args,
        )

        train_ds = NLUdataset(texts_train, intents_train, entities_train)
        test_ds = NLUdataset(texts_test, intents_test, entities_test)
        return train_ds, test_ds

    def _filter_by_index_list(self, idx_list):
        filtered = [d for (idx, d) in enumerate(self.data) if idx in idx_list]
        texts, intents, entities = zip(*filtered)
        return NLUdataset(texts, intents, entities)

    def cross_validation_splits(self, cv_iterator=None):
        """Cross validation generator function."""
        if cv_iterator is None:
            cv_iterator = StratifiedKFold(n_splits=5, shuffle=True)
        for train_idx, test_idx in cv_iterator.split(self.texts, self.intents):
            train_ds = self._filter_by_index_list(train_idx)
            test_ds = self._filter_by_index_list(test_idx)
            yield train_ds, test_ds

    def to_records(self):
        """
        Return data as a list of dicts.

        Each dict represents a single utterance/document.
        """
        examples = []
        for text, intent, entities in self.data:
            example = {
                "text": text,
                "intent": intent,
                "entities": [
                    {
                        "entity": entity[EntityKeys.TYPE],
                        "start": entity[EntityKeys.START],
                        "end": entity[EntityKeys.END],
                    }
                    for entity in entities
                ],
            }  # yapf: disable
            examples.append(example)
        return examples

    def to_dict(self):
        """
        Return intent data as dict.

        Intents are keys, each holding a list of corresponding
        utterances.
        """
        d = collections.defaultdict(list)
        for text, intent, _ in self.data:
            d[intent].append(text)

        return dict(d)

    def to_json(self, path=None, records=None):
        """
        Convert dataset to JSON.

        If a path is given, saves to that file. Otherwise returns a
        JSONstring. Records can be passed explicitly, otherwise
        self.to_records() will be called to obtain the data in
        dictionary format.

        :param path: optional path under which to save the JSON file
        :type path: str
        :param records: optional JSON-serializable data
        :type records: dict or list
        """
        if records is None:
            records = self.to_records()

        if path is None:
            return json.dumps(records, indent=4, ensure_ascii=False)

        with open(path, "w") as f:
            json.dump(records, f, indent=4, ensure_ascii=False)
