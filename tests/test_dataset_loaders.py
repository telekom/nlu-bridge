import os

from test_datasets import FIXTURE_PATH


def test_from_huggingface():
    # TODO: Figure out how to easily test this with a fixture hugging_ds
    pass


def test_from_watson():
    from nlubridge.vendors.watson import load_data

    ds = load_data(os.path.join(FIXTURE_PATH, "watson_intents_export.csv"))
    assert len(ds) == 4
    assert ds.texts[0][0] != '"'
    assert ds.texts[2] == "testing umläuts"
    assert ds.intents[0] == "intent1"
    assert ds.intents[1] == "intent2"


def test_from_rasa_json():
    from nlubridge.vendors.rasa import load_data

    ds = load_data(os.path.join(FIXTURE_PATH, "rasa_nlu.json"))
    assert len(ds) == 5
    assert ds.texts[1] == "testing umläuts"
    assert ds.intents[0] == "restaurant_search"
    assert ds.intents[3] == "affirm"
    assert ds.entities[0] == []
    assert ds.entities[1] == [
        {"start": 31, "end": 36, "value": "north", "entity": "location"}
    ]
    assert len(ds.entities[4]) == 2
    # make sure the custom "role" key is in dataset
    assert ds.entities[4][0].get("role", False)
    # check assumptions about indexes hold
    idx1 = ds.entities[4][0]["start"]
    idx2 = ds.entities[4][0]["end"]
    value = ds.entities[4][0]["value"]
    assert ds.texts[4][idx1:idx2] == value


def test_from_luis():
    from nlubridge.vendors.luis import load_data

    ds = load_data(os.path.join(FIXTURE_PATH, "luis_nlu.json"))
    assert len(ds) == 5
    assert ds.texts[1] == "testing umläuts"
    assert ds.intents[0] == "restaurant_search"
    assert ds.intents[3] == "affirm"
    assert ds.entities[0] == []
    assert ds.entities[1] == [
        {"start": 31, "end": 36, "value": "north", "entity": "location"}
    ]
    assert len(ds.entities[4]) == 2
    # make sure the custom "role" key is in dataset
    assert ds.entities[4][0].get("role", False)
    # check assumptions about indexes hold
    idx1 = ds.entities[4][0]["start"]
    idx2 = ds.entities[4][0]["end"]
    value = ds.entities[4][0]["value"]
    assert ds.texts[4][idx1:idx2] == value
