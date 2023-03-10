from packaging import version
from test_datasets import entities as sample_entities
from test_datasets import intents as sample_intents
from test_datasets import texts as sample_texts

from nlubridge import EntityKeys, NBestKeys, NluDataset


# Following functions are run by all vendor tests to test
# functionalities shared by all
# -----------------------------


def assert_preds_are_intents(vendor, unique_intents):
    test_ds = NluDataset(["Ich habe kein DSL und telefon"])
    preds = vendor.test_intent(test_ds)
    assert isinstance(preds, list)
    assert len(preds) == 1
    assert preds[0] in unique_intents


def assert_return_probs(vendor, unique_intents):
    test_ds = NluDataset(["Ich habe kein DSL und telefon"])
    preds, probs = vendor.test_intent(test_ds, return_probs=True)
    assert preds[0] in unique_intents
    assert isinstance(probs[0], float) or (probs[0] == 1)


def assert_multiple_utterances_predicted(vendor, train_data):
    preds, probs = vendor.test_intent(train_data[:10], return_probs=True)
    assert len(preds) == 10
    assert isinstance(preds[0], str)
    assert len(probs) == 10
    assert isinstance(probs[0], float) or (probs[0] == 1)


# Following are tests for each vendor. We bundled test by vendor so we
# can test an individual vendor independently during development
# -----------------------------


def test_vendor_attributes():
    from nlubridge.vendors.tfidf_intent_classifier import TfidfIntentClassifier

    clf = TfidfIntentClassifier()
    assert clf.name == "TfidfIntentClassifier"
    assert clf.alias == "TfidfIntentClassifier"
    clf.alias = "tfidf"
    assert clf.alias == "tfidf"


def test_tfidf(train_data):
    from nlubridge.vendors.tfidf_intent_classifier import TfidfIntentClassifier

    # test initialization
    bow = TfidfIntentClassifier()

    # test train_intent()
    bow.train_intent(train_data)

    # test test_intent()
    assert_preds_are_intents(bow, train_data.unique_intents)
    assert_return_probs(bow, train_data.unique_intents)
    assert_multiple_utterances_predicted(bow, train_data)


def test_rasa(train_data):
    from rasa import __version__ as rasa_version

    if version.parse(rasa_version) < version.parse("3.0.0"):
        from nlubridge.vendors.rasa2 import Rasa2

        rasa = Rasa2()
    else:
        from nlubridge.vendors.rasa3 import Rasa3

        rasa = Rasa3()

    # test intent classification
    rasa.train_intent(train_data)
    assert_preds_are_intents(rasa, train_data.unique_intents)
    assert_return_probs(rasa, train_data.unique_intents)
    assert_multiple_utterances_predicted(rasa, train_data)

    # test intent + entity classification
    train_ds = NluDataset(sample_texts, sample_intents, sample_entities)
    rasa.train(train_ds)
    test_ds = NluDataset(
        ["I want a flight from Frankfurt to Berlin", "How is the weather in Bonn?"]
    )
    preds = rasa.test(test_ds)
    assert isinstance(preds, NluDataset)
    assert len(preds) == 2
    assert len(preds.confidences) == 2
    assert isinstance(preds.confidences[0], float)
    assert len(preds.n_best_intents) == 2
    assert isinstance(preds.n_best_intents[0], list)
    assert isinstance(preds.n_best_intents[0][0], dict)
    assert preds.n_best_intents[0][0][NBestKeys.INTENT] in train_ds.unique_intents
    assert len(preds.entities) == 2
    assert isinstance(preds.entities[0], list)
    assert isinstance(preds.entities[0][0], dict)
    assert preds.entities[0][0][EntityKeys.TYPE] in train_ds.unique_entities


def test_telekom(train_data):
    from nlubridge.vendors.char_ngram_intent_classifier import CharNgramIntentClassifier

    model = CharNgramIntentClassifier()
    model.train_intent(train_data)
    assert_preds_are_intents(model, train_data.unique_intents)
    assert_return_probs(model, train_data.unique_intents)
    assert_multiple_utterances_predicted(model, train_data)
    # TODO: fix vendor for test: assert_oos_prediction(mediaan)


def test_spacy(train_data):
    from nlubridge.vendors.spacy import Spacy

    # We train with small number of train iterations to speed up tests
    # (performance not important here)
    model = Spacy(n_iter=10)
    model.train_intent(train_data)
    predicted = model.test_intent(train_data)
    assert len(predicted) == len(train_data)
    # assert_preds_are_intents(spacy, train_data.unique_intents)
    # assert_return_probs(spacy, train_data.unique_intents)
    # assert_multiple_utterances_predicted(spacy, train_data)
    # assert_oos_prediction(spacy)


def test_fasttext(train_data):
    from nlubridge.vendors.fasttext import FastText

    fasttext = FastText(epochs=10)
    fasttext.train_intent(train_data)
    assert_preds_are_intents(fasttext, train_data.unique_intents)
    assert_return_probs(fasttext, train_data.unique_intents)
    assert_multiple_utterances_predicted(fasttext, train_data)
