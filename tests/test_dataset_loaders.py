import json
import os

from test_datasets import FIXTURE_PATH


def test_from_huggingface_intents():
    from datasets import ClassLabel, Dataset

    from nlubridge import from_huggingface

    with open(
        os.path.join(FIXTURE_PATH, "huggingface_banking77.json"), "r", encoding="utf-8"
    ) as f:
        testdata = json.load(f)

    hf_ds = Dataset.from_dict(testdata["data"])
    hf_ds.features["label"] = ClassLabel(
        num_classes=len(testdata["label_class_names"]),
        names=testdata["label_class_names"],
    )
    ds = from_huggingface(hf_ds, has_intents=True, has_entities=False)

    assert len(ds) == 8
    assert len(ds.intents) == len(ds)
    assert all(len(x) > 0 for x in ds.intents)
    assert len(ds.texts) == len(ds)
    assert all(len(x) > 0 for x in ds.texts)
    assert len(ds.entities) == len(ds)
    assert all(len(x) == 0 for x in ds.entities)
    assert ds.texts[0] == "I am still waiting on my card?"
    assert ds.texts[3] == "What is my money worth in other countries?"
    assert ds.intents[0] == "card_arrival"
    assert ds.intents[3] == "exchange_rate"


def test_from_huggingface_entities():
    from datasets import ClassLabel, Dataset

    from nlubridge import from_huggingface

    with open(
        os.path.join(FIXTURE_PATH, "huggingface_wnut_17.json"), "r", encoding="utf-8"
    ) as f:
        testdata = json.load(f)

    hf_ds = Dataset.from_dict(testdata["data"])
    hf_ds.features["ner_tags"].feature = ClassLabel(
        num_classes=len(testdata["ner_tags_class_names"]),
        names=testdata["ner_tags_class_names"],
    )

    ds = from_huggingface(hf_ds, has_intents=False, has_entities=True)

    assert len(ds) == 8
    assert len(ds.intents) == len(ds)
    assert all(x is None for x in ds.intents)
    assert len(ds.texts) == len(ds)
    assert all(len(x) > 0 for x in ds.texts)
    assert len(ds.entities) == len(ds)
    assert all(len(x) >= 0 for x in ds.entities)
    assert ds.texts[3] == "today is my last day at the office."
    assert len(ds.entities[0]) == 2
    assert ds.entities[3] == []
    assert ds.entities[4] == [{"start": 0, "end": 7, "entity": "person"}]
    assert ds.entities[6] == [{"start": 18, "end": 43, "entity": "product"}]


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
    assert ds.entities[3][1].get("role", False)
    # check assumptions about indexes hold
    idx1 = ds.entities[3][1]["start"]
    idx2 = ds.entities[3][1]["end"]
    value = ds.entities[3][1]["value"]
    assert ds.texts[3][idx1:idx2] == value


def test_from_luis():
    from nlubridge import from_luis

    ds = from_luis(os.path.join(FIXTURE_PATH, "luis_nlu.json"))
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
