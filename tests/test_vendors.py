import pytest

from dotenv import load_dotenv

from nlubridge import NLUdataset
from testing_data import TrainingDataset


@pytest.fixture
def train_data():
    return TrainingDataset()


#    return TDGDataset().clip_by_intent_frequency(20)


# Following functions are run by all vendor tests to test
# functionalities shared by all
# -----------------------------


def assert_preds_are_intents(vendor, unique_intents):
    test_ds = NLUdataset(["Ich habe kein DSL und telefon"])
    preds = vendor.test_intent(test_ds)
    assert isinstance(preds, list)
    assert len(preds) == 1
    assert preds[0] in unique_intents


def assert_return_probs(vendor, unique_intents):
    test_ds = NLUdataset(["Ich habe kein DSL und telefon"])
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


def test_watson(train_data):
    from nlubridge.vendors.watson import Watson

    # load environment variables so Watson uses them
    load_dotenv()

    # test initialization
    watson = Watson()

    # test train_intent()
    watson.train_intent(train_data)

    # test test_intent() (bulk_classify)
    assert_preds_are_intents(watson, train_data.unique_intents)
    assert_return_probs(watson, train_data.unique_intents)
    assert_multiple_utterances_predicted(watson, train_data)
    # test handling of rate exceeded (only 250/min allowed)
    cases90 = train_data + train_data + train_data
    cases360 = cases90 + cases90 + cases90 + cases90
    preds, probs = watson.test_intent(cases360, return_probs=True)
    assert len(preds) == 360
    assert len(probs) == 360
    # TODO: fix vendor for test: assert_oos_prediction()

    # test test_intent() (non-bulk)
    test_ds = NLUdataset(["Ich habe kein DSL und telefon"])
    watson.set_bulk(False)
    assert watson.use_bulk is False
    preds = watson.test_intent(test_ds)
    assert isinstance(preds, list)
    assert len(preds) == 1
    assert preds[0] in train_data.unique_intents

    # test n_best_intents argument to test_intent()
    preds, probs = watson.test_intent(test_ds, return_probs=True, n_best_intents=5)
    assert isinstance(preds, list)
    assert isinstance(probs, list)
    assert isinstance(preds[0], list)
    assert isinstance(probs[0], list)
    assert preds[0][0] in train_data.unique_intents
    assert len(preds[0]) == 2
    assert isinstance(probs[0][0], float) or (probs[0][0] == 1)
    assert len(probs[0]) == 2


def test_rasa(train_data):
    from nlubridge.vendors.rasa import Rasa

    rasa = Rasa()
    rasa.train_intent(train_data)
    assert_preds_are_intents(rasa, train_data.unique_intents)
    assert_return_probs(rasa, train_data.unique_intents)
    assert_multiple_utterances_predicted(rasa, train_data)


def test_telekom(train_data):
    from nlubridge.vendors.telekom import TelekomModel

    model = TelekomModel()
    model.train_intent(train_data)
    assert_preds_are_intents(model, train_data.unique_intents)
    assert_return_probs(model, train_data.unique_intents)
    assert_multiple_utterances_predicted(model, train_data)
    # TODO: fix vendor for test: assert_oos_prediction(mediaan)


def test_spacy(train_data):
    from nlubridge.vendors.spacy import SpacyClassifier

    # We train with small number of train iterations to speed up tests
    # (performance not important here)
    model = SpacyClassifier(n_iter=10)
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
