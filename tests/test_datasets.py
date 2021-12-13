from collections import Counter
import os
import copy

import pytest

from testing_data import SyntheticDataset, ToyDataset
from nlubridge import OUT_OF_SCOPE_TOKEN, NLUdataset
from sklearn.model_selection import train_test_split, KFold

FIXTURE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures")


texts = [
    "Book me a flight from Cairo to Redmond next Thursday",
    "What's the weather like in Seattle?",
    "What's the weather in Berlin?",
]
intents = ["BookFlight", "GetWeather", "GetWeather"]
entities = [
    [
        {"entity": "Location::From", "start": 22, "end": 26},
        {"entity": "Location::To", "start": 31, "end": 37},
    ],
    [{"entity": "Location", "start": 27, "end": 33}],
    [{"entity": "Location", "start": 23, "end": 27}],
]


def test_init_ds_with_texts_intents_entities():
    ds = NLUdataset(texts, intents, entities)
    assert ds.texts == texts
    assert ds.intents == intents
    assert ds.unique_intents == ["BookFlight", "GetWeather"]
    assert ds.n_intents == 2
    assert ds.entities == entities
    assert ds.unique_entities == ["Location::From", "Location::To", "Location"]
    assert ds.n_entities == 3


def test_crop_intent_names():
    ds = NLUdataset(texts, intents, entities, max_intent_length=4)
    assert ds.texts == texts
    assert ds.intents == ["Book", "GetW", "GetW"]
    assert ds.unique_intents == ["Book", "GetW"]
    assert ds.n_intents == 2
    # check a warning is thrown when cropping makes intent names ambiguous
    with pytest.warns(UserWarning):
        ds2 = NLUdataset(
            texts + ["bla"],
            intents + ["GetWhatever"],
            entities + [[]],
            max_intent_length=4,
        )
    assert ds2.n_intents == 2


def test_out_of_scope():
    ds1 = NLUdataset(texts, intents, entities)
    ds2 = NLUdataset(texts, intents, entities, out_of_scope=True)

    assert ds2.n_intents == ds1.n_intents + 1
    assert OUT_OF_SCOPE_TOKEN in ds2.unique_intents


def test_init_ds_without_intents_and_entities():
    ds = NLUdataset(texts)
    assert len(ds.texts) == 3
    assert len(ds.intents) == 3
    assert ds.unique_intents is None  # NOTE: This should be [] for consistency
    assert ds.intents == [None, None, None]
    assert ds.n_intents is None  # NOTE: This should be [] for consistency
    assert len(ds.entities) == 3
    assert ds.unique_entities == []
    assert ds.entities == [[], [], []]
    assert ds.n_entities == 0


def test_ds_unique():
    ds = ToyDataset()
    assert ds.n_intents == 2
    assert ds.n_entities == 3


# def test_ds_from_json():
#     with open("data/simple_dataset.json") as f:
#         test_json = f.read()
#     dataset = NLUdataset.from_json(test_json)
#     assert len(dataset.intents) == 2


def test_ds_slicing():
    ds = NLUdataset(texts, intents, entities)
    assert len(ds[:2]) == 2


def test_ds_slicing_when_created_with_just_texts():
    # Ensure that when we pass a list of Nones for intents (during
    # construction of returned dataset in slicing) the returned
    # dataset is still valid
    ds = NLUdataset(texts)
    sliced = ds[:2]
    assert len(sliced.texts) == 2
    assert len(sliced.intents) == 2
    assert len(sliced.entities) == 2
    assert len(sliced) == 2


def test_ds_from_joined():
    ds1 = ToyDataset()
    ds2 = ToyDataset()
    joined = NLUdataset.from_joined(ds1, ds2)
    assert joined.n_samples == ds1.n_samples + ds2.n_samples


def test_to_records():
    ds = ToyDataset()
    records = ds.to_records()
    assert len(records) == ds.n_samples
    assert isinstance(ds.to_json(), str)


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
    with pytest.raises(NotImplementedError):
        ds.subsample_by_intent_frequency(0.5, 1500, shuffle=True)


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

    from sklearn.model_selection import train_test_split

    train, test = train_test_split(ds, test_size=0.3)
    assert len(train) == 7
    assert len(test) == 3


def test__filter_by_index_list():
    pass


def test_cross_validation_splits():
    pass


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
    ds1 = ds.sample(1000)
    ds2 = ds.sample(1000)
    assert ds1.texts[0] == ds2.texts[0]

    # ds3 = ds1.sample(100)
    # assert ds1.texts[0] != ds3.texts[0]


def test_validation():
    ds = SyntheticDataset(10, intents=["intent1", "intent2"])

    ds_train, ds_test = train_test_split(ds, test_size=0.2)
    assert len(ds_train) == 8

    kf = KFold(n_splits=2)
    for train_idx, test_idx in kf.split(ds):
        train_ds, test_ds = ds[train_idx], ds[test_idx]
        assert len(train_ds.texts) == len(train_ds.intents)
        assert len(test_ds.texts) == len(test_ds.intents)
