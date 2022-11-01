import os

from test_datasets import FIXTURE_PATH


def test_from_huggingface():
    # TODO: Figure out how to easily test this with a fixture hugging_ds
    pass


def test_from_watson():
    from nlubridge import from_watson_assistant

    ds = from_watson_assistant(os.path.join(FIXTURE_PATH, "watson_intents_export.csv"))
    assert len(ds) == 4
    assert ds.texts[0][0] != '"'
    assert ds.texts[2] == "testing umläuts"
    assert ds.intents[0] == "intent1"
    assert ds.intents[1] == "intent2"


def test_from_rasa_json():
    from nlubridge import from_rasa

    ds = from_rasa(os.path.join(FIXTURE_PATH, "rasa_nlu.json"), format="json")
    assert len(ds) == 5
    assert ds.texts[1] == "testing umläuts"
    assert ds.intents[0] == "restaurant_search"
    assert ds.intents[3] == "affirm"
    assert ds.entities[0] == []
    assert len(ds.entities[1]) == 1
    assert ds.entities[1][0].as_dict() == {
        "start": 31,
        "end": 36,
        "value": "north",
        "entity": "location",
    }
    assert len(ds.entities[4]) == 2
    # make sure the custom "role" key is in dataset
    assert ds.entities[4][0].data.get("role", False)
    # check assumptions about indexes hold
    idx1 = ds.entities[4][0].start
    idx2 = ds.entities[4][0].end
    value = ds.entities[4][0].data["value"]
    assert ds.texts[4][idx1:idx2] == value


def test_from_rasa3_yml():
    from nlubridge import from_rasa

    ds = from_rasa(os.path.join(FIXTURE_PATH, "rasa3_nlu.yml"))
    assert len(ds) == 5
    assert ds.texts[1] == "testing umläuts"
    assert ds.intents[0] == "restaurant_search"
    assert ds.intents[4] == "affirm"
    assert ds.entities[0] == []
    assert ds.entities[1] == []
    assert len(ds.entities[3]) == 2
    # make sure the custom "role" key is in dataset
    assert ds.entities[3][1].data.get("role", False)
    # check assumptions about indexes hold
    idx1 = ds.entities[3][1].start
    idx2 = ds.entities[3][1].end
    value = ds.entities[3][1].data["value"]
    assert ds.texts[3][idx1:idx2] == value


def test_from_luis():
    from nlubridge import from_luis

    ds = from_luis(os.path.join(FIXTURE_PATH, "luis_nlu.json"))
    assert len(ds) == 5
    assert ds.texts[1] == "testing umläuts"
    assert ds.intents[0] == "restaurant_search"
    assert ds.intents[3] == "affirm"
    assert ds.entities[0] == []
    assert len(ds.entities[1]) == 1
    assert ds.entities[1][0].as_dict() == {
        "start": 31,
        "end": 36,
        "value": "north",
        "entity": "location",
    }

    assert len(ds.entities[4]) == 2
    # make sure the custom "role" key is in dataset
    assert ds.entities[4][0].data.get("role", False)
    # check assumptions about indexes hold
    idx1 = ds.entities[4][0].start
    idx2 = ds.entities[4][0].end
    value = ds.entities[4][0].data["value"]
    assert ds.texts[4][idx1:idx2] == value
