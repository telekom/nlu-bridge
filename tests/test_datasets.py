import copy
import os
from collections import Counter

import pytest
from sklearn.model_selection import KFold
from testing_data import SyntheticDataset, ToyDataset

from nlubridge import EntityKeys, NBestKeys, NluDataset, concat, from_json


FIXTURE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures")


texts = [
    "Book me a flight from Cairo to Redmond next Thursday",
    "What's the weather like in Seattle?",
    "What's the weather in Berlin?",
]
intents = ["BookFlight", "GetWeather", "GetWeather"]
entities = [
    [
        {EntityKeys.TYPE: "Location::From", EntityKeys.START: 22, EntityKeys.END: 27},
        {EntityKeys.TYPE: "Location::To", EntityKeys.START: 31, EntityKeys.END: 38},
    ],
    [{EntityKeys.TYPE: "Location", EntityKeys.START: 27, EntityKeys.END: 34}],
    [{EntityKeys.TYPE: "Location", EntityKeys.START: 22, EntityKeys.END: 28}],
]
n_best_intents = [
    [
        {NBestKeys.INTENT: "BookFlight", NBestKeys.CONFIDENCE: 0.5},
        {NBestKeys.INTENT: "GetWeather", NBestKeys.CONFIDENCE: 0.4},
    ],
    [
        {NBestKeys.INTENT: "GetWeather", NBestKeys.CONFIDENCE: 0.9},
        {NBestKeys.INTENT: "BookFlight", NBestKeys.CONFIDENCE: 0.1},
    ],
    [
        {NBestKeys.INTENT: "BookFlight", NBestKeys.CONFIDENCE: 0.8},
        {NBestKeys.INTENT: "GetWeather", NBestKeys.CONFIDENCE: 0.2},
    ],
]


def test_init_ds_with_texts_intents_entities():
    ds = NluDataset(texts, intents, entities)
    assert ds.texts == texts
    assert ds.intents == intents
    assert ds.unique_intents == ["BookFlight", "GetWeather"]
    assert ds.n_intents == 2
    assert ds.entities == entities
    assert ds.unique_entities == ["Location", "Location::From", "Location::To"]
    assert ds.n_entities == 3


def test_init_from_data():
    _ds = NluDataset(texts, intents, entities)
    ds = NluDataset._from_data(_ds._data)
    assert ds.texts == texts
    assert ds.intents == intents
    assert ds.unique_intents == ["BookFlight", "GetWeather"]
    assert ds.n_intents == 2
    assert ds.entities == entities
    assert ds.unique_entities == ["Location", "Location::From", "Location::To"]
    assert ds.n_entities == 3


def test_crop_intent_names():
    ds = NluDataset(texts, intents, entities, max_intent_length=4)
    assert ds.texts == texts
    assert ds.intents == ["Book", "GetW", "GetW"]
    assert ds.unique_intents == ["Book", "GetW"]
    assert ds.n_intents == 2
    # check a warning is thrown when cropping makes intent names ambiguous
    with pytest.warns(UserWarning):
        ds2 = NluDataset(
            texts + ["bla"],
            intents + ["GetWhatever"],
            entities + [[]],
            max_intent_length=4,
        )
    assert ds2.n_intents == 2


def test_init_ds_without_intents_and_entities():
    ds = NluDataset(texts)
    assert len(ds.texts) == 3
    assert len(ds.intents) == 3
    assert ds.unique_intents == []
    assert ds.intents == [None, None, None]
    assert ds.n_intents == 0
    assert ds.intent_frequencies == Counter({None: 3})
    assert len(ds.entities) == 3
    assert ds.unique_entities == []
    assert ds.entities == [[], [], []]
    assert ds.n_entities == 0
    assert ds.confidences == [None, None, None]
    assert ds.n_best_intents == [[], [], []]


def test_init_with_nbest_list():
    ds = NluDataset(texts, intents, entities, n_best_intents)
    assert len(ds.n_best_intents) == 3
    assert ds.confidences == [0.5, 0.9, 0.8]
    assert ds._data[0][3] == n_best_intents[0]


def test_clip_by_intent_frequency():
    pass


def test__filter_by_index_list():
    pass


def test_ds_unique():
    ds = ToyDataset()
    assert ds.n_intents == 2
    assert ds.n_entities == 3


def test_ds_slicing():
    ds = NluDataset(texts, intents, entities)
    ds_sliced = ds[:2]
    assert isinstance(ds_sliced, NluDataset)
    assert len(ds_sliced) == 2
    ds_sliced = ds[2]
    assert isinstance(ds_sliced, NluDataset)
    assert len(ds_sliced) == 1


def test_ds_slicing_when_created_with_just_texts():
    # Ensure that when we pass a list of Nones for intents (during
    # construction of returned dataset in slicing) the returned
    # dataset is still valid
    ds = NluDataset(texts)
    sliced = ds[:2]
    assert len(sliced.texts) == 2
    assert len(sliced.intents) == 2
    assert len(sliced.entities) == 2
    assert len(sliced) == 2


def test_cross_validation_splits():
    ds = SyntheticDataset(10, intents=["intent1", "intent2"])
    for ds_train, ds_test in ds.cross_validation_splits():
        assert isinstance(ds_train, NluDataset)
        assert isinstance(ds_test, NluDataset)
        assert len(ds_train.texts) == 8
        assert len(ds_test.texts) == 2
        assert len(ds_train.texts) == len(ds_train.intents)
        assert len(ds_test.texts) == len(ds_test.intents)

    # Also test for leave-1-out (special case because the dataset has only a single
    # record)
    kf = KFold(n_splits=10)
    for ds_train, ds_test in ds.cross_validation_splits(kf):
        assert isinstance(ds_train, NluDataset)
        assert isinstance(ds_test, NluDataset)
        assert len(ds_train.texts) == 9
        assert len(ds_test.texts) == 1
        assert len(ds_train.texts) == len(ds_train.intents)
        assert len(ds_test.texts) == len(ds_test.intents)


def test_ds_from_joined():
    ds1 = ToyDataset()
    ds2 = ToyDataset()
    with pytest.warns(DeprecationWarning):
        joined = NluDataset.from_joined(ds1, ds2)
    assert joined.n_samples == ds1.n_samples + ds2.n_samples


def test_ds_join():
    ds1 = ToyDataset()
    ds2 = ToyDataset()
    ds1_samples = ds1.n_samples
    ds2_samples = ds2.n_samples
    joined = ds1.join(ds2)
    assert joined.n_samples == ds1_samples + ds2_samples
    # assert that the original dataset was not changed
    assert ds1_samples == ds1.n_samples
    joined2 = ds2.join(ds1)
    assert joined2.n_samples == ds1.n_samples + ds2.n_samples


def test_nlubridge_concat():
    ds1 = ToyDataset()
    ds2 = ToyDataset()
    ds3 = ToyDataset()
    ds1_samples = ds1.n_samples
    ds2_samples = ds2.n_samples
    ds3_samples = ds3.n_samples
    joined = concat([ds1, ds2, ds3])
    assert joined.n_samples == ds1_samples + ds2_samples + ds3_samples
    assert ds1_samples == ds1.n_samples
    assert ds2_samples == ds2.n_samples
    assert ds3_samples == ds3.n_samples


def test_to_records():
    ds = ToyDataset()
    with pytest.warns(DeprecationWarning):
        examples = ds.to_records()
    assert len(examples) == ds.n_samples


def test_to_json():
    ds = ToyDataset()
    examples = ds.to_json()
    assert isinstance(examples, list)
    assert isinstance(examples[0], dict)
    assert len(examples) == ds.n_samples


def test_to_json_from_json():
    ds = ToyDataset()
    examples = ds.to_json()
    ds2 = from_json(examples=examples)
    assert ds.texts == ds2.texts
    assert ds.intents == ds2.intents
    assert ds.entities == ds2.entities


def test_filter_by_intent_name():
    intents = [
        "intent_1",
        "intent_2",
        "intent_3",
        "intent_4",
        "intent_5",
        "intent_6",
    ]
    ds = SyntheticDataset(n_samples=100, intents=intents)
    excluded = ["intent_2", "intent_4"]
    filtered = ds.filter_by_intent_name(excluded=excluded)
    expected = [
        "intent_1",
        "intent_3",
        "intent_5",
        "intent_6",
    ]
    assert filtered.unique_intents == expected


# def test_select_first_n_intents():
#     # ds = select_first_n_intents(self, 2)
#     pass


def get_subsampling_ds():
    intents = (
        ["intent_1"] * 3000
        + ["intent_2"] * 3000
        + ["intent_3"] * 4500
        + ["intent_4"] * 500
    )
    ds = SyntheticDataset(n_samples=len(intents), intents=intents)
    return ds


def test_subsample_by_intent_frequency():
    ds = get_subsampling_ds()
    ds_sub = ds.subsample_by_intent_frequency(0.5, 1500)
    counter = Counter(ds_sub.intents)
    assert counter["intent_1"] == 2250
    assert counter["intent_2"] == 2250
    assert counter["intent_3"] == 3000
    assert counter["intent_4"] == 500
    # ensure intents and texts are not mixed up
    pairs = set(zip(ds.texts, ds.intents))
    pairs_sub = set(zip(ds_sub.texts, ds_sub.intents))
    assert len(pairs_sub - pairs) == 0


def test_subsample_by_intent_frequency_rate_is_zero():
    ds = get_subsampling_ds()
    ds_sub = ds.subsample_by_intent_frequency(0, 1500)
    counter = Counter(ds_sub.intents)
    assert counter["intent_1"] == 1500
    assert counter["intent_2"] == 1500
    assert counter["intent_3"] == 1500
    assert counter["intent_4"] == 500


def test_subsample_by_intent_frequency_rate_is_one():
    ds = get_subsampling_ds()
    ds_sub = ds.subsample_by_intent_frequency(1, 1500)
    counter = Counter(ds_sub.intents)
    assert counter["intent_1"] == 3000
    assert counter["intent_2"] == 3000
    assert counter["intent_3"] == 4500
    assert counter["intent_4"] == 500


def test_shuffle():
    ds = SyntheticDataset(n_samples=10, intents=["a", "b", "c"])
    shuffled = copy.deepcopy(ds).shuffle()
    assert ds.n_samples == shuffled.n_samples
    assert set(ds.texts) == set(shuffled.texts)
    assert ds.texts != shuffled.texts


def test_train_test_split():
    ds = SyntheticDataset(n_samples=10, intents=["a", "b", "c"])
    train, test = ds.train_test_split(test_size=0.3)
    assert train.n_samples == 7
    assert test.n_samples == 3


def test_cv_splits_texts_are_list():
    # addresses a former error where in the cv method tuples were
    # passed and to the dataset constructor for the partitioning
    ds = SyntheticDataset(n_samples=20, intents=["a", "b", "c"])
    for train_ds, test_ds in ds.cross_validation_splits():
        assert isinstance(train_ds.texts, list)
        assert isinstance(train_ds.intents, list)
        assert isinstance(train_ds.entities, list)
        assert isinstance(test_ds.texts, list)


def test_sample():
    size = 0.1
    ds = SyntheticDataset(n_samples=2000, intents=["a", "b", "c"])
    sampled = ds.sample(size=size)
    assert ds.n_intents == sampled.n_intents

    ratio = sampled.n_samples / ds.n_samples
    tolerance = 0.001
    assert (ratio - size) < tolerance

    # test reproduciblity
    ds1 = ds.sample(1000, random_state=42)
    ds2 = ds.sample(1000, random_state=42)
    assert ds1.texts[0] == ds2.texts[0]

    # ds3 = ds1.sample(100)
    # assert ds1.texts[0] != ds3.texts[0]
