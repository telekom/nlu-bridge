from dotenv import load_dotenv
from test_vendors import (
    assert_multiple_utterances_predicted,
    assert_preds_are_intents,
    assert_return_probs,
)

from nlubridge import NluDataset


def test_watson(train_data):
    from nlubridge.vendors.watson_assistant import WatsonAssistant

    # load environment variables so Watson uses them
    load_dotenv()

    # test initialization
    watson = WatsonAssistant()

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
    test_ds = NluDataset(["Ich habe kein DSL und telefon"])
    watson.set_bulk(False)
    assert watson._use_bulk is False
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
