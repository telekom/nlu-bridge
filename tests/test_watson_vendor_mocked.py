import json

import requests
from test_vendors import (
    assert_multiple_utterances_predicted,
    assert_preds_are_intents,
    assert_return_probs,
)

from nlubridge import NluDataset
from nlubridge.vendors.watson_assistant import WatsonAssistant


FAKE_URL = "dummy.org"
FAKE_KEY = "dummy_key"
WS_NAME = "dummy_name"
WS_ID = 1


def test_watson_mocked(train_data, mocker):
    # TODO: Also mock tests for use_bulk=False (see test_watson_vendor.py)

    # Mock the ibm_watson AssistantV1 API for requests implemented by this
    mocker.patch("nlubridge.vendors.watson_assistant.AssistantV1", AssitantV1Mock)

    # Mock requests.Session.post() requests
    mocker.patch("requests.Session", SessionMock)

    # test initialization
    watson = WatsonAssistant(FAKE_URL, FAKE_KEY, WS_NAME)

    # test train_intent()
    watson.train_intent(train_data)

    # test test_intent() and properties of its return values
    assert_preds_are_intents(watson, train_data.unique_intents)
    assert_return_probs(watson, train_data.unique_intents)
    assert_multiple_utterances_predicted(watson, train_data)

    # Test for datasets that need to be passed in multiple batches
    # TODO: test handling of rate-limit (only 250/min allowed)
    cases90 = train_data + train_data + train_data
    cases360 = cases90 + cases90 + cases90 + cases90
    preds, probs = watson.test_intent(cases360, return_probs=True)
    assert len(preds) == 360
    assert len(probs) == 360

    test_ds = NluDataset(["Ich habe kein DSL und telefon"])

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

    # test watson.set_bulk()
    watson.set_bulk(False)
    assert watson._use_bulk is False


class ResultMock:
    """
    We sometimes need to be able to call get_result() on a return value from
    AssistantV1Mock.
    """

    def __init__(self, result, status_code=None):
        self.result = result
        self.status_code = status_code

    def get_result(self):
        return self.result

    def get_status_code(self):
        return self.result

    def json(self):
        return self.result


class TokenManager:
    apikey = FAKE_KEY


class Authenticator:
    token_manager = TokenManager()


class AssitantV1Mock:
    """
    Mocks AssistantV1 from the ibm_watson package
    """

    service_url = FAKE_URL
    version = "111"
    authenticator = Authenticator()
    workspace_name = WS_NAME

    def __init__(self, **kwargs):
        return None

    @staticmethod
    def set_service_url(endpoint):
        pass

    @staticmethod
    def get_workspace(workspace_id, export):
        return ResultMock({"status": "Available"})

    @staticmethod
    def list_workspaces():
        result = {
            "workspaces": [
                {"name": WS_NAME, "workspace_id": WS_ID},
                {"name": "workspace2", "workspace_id": "2"},
            ]
        }
        return ResultMock(result)

    @staticmethod
    def delete_workspace(id):
        return ResultMock(200)

    @staticmethod
    def create_workspace(name, description, language):
        return ResultMock({"workspace_id": WS_ID})

    @staticmethod
    def create_intent(workspace_id, intent, examples):
        return ResultMock(None)


class SessionMock(requests.Session):
    """
    Mocks requests.Session.post()
    """

    def post(self, url, data):
        data_dict = json.loads(data)
        texts = [item["text"] for item in data_dict["input"]]
        return ResultMock(
            {
                "output": [
                    {
                        "input": {"text": text},
                        "entities": [],
                        "intents": [
                            {"intent": "help", "confidence": 0.061},
                            {"intent": "affirm", "confidence": 0.0448},
                        ],
                    }
                    for text in texts
                ]
            },
            200,
        )
