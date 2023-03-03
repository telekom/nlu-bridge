# Copyright (c) 2021 Yaser Martinez-Palenzuela, Deutsche Telekom AG
# This software is distributed under the terms of the MIT license
# which is available at https://opensource.org/licenses/MIT

import concurrent.futures
import datetime
import json
import logging
import os
import sys
import time

import requests
from ratelimit import rate_limited

from ..nlu_dataset import OUT_OF_SCOPE_TOKEN
from .vendor import Vendor


sys.path.append("../..")
logger = logging.getLogger(__name__)
global n_calls
global start_time


def _unwrap_self(arg, **kwarg):
    return Luis.test_single_intent(*arg, **kwarg)


class Luis(Vendor):
    AUTHORING_RATE_LIMIT = 4.99  # queries per second
    SUBSCRIPTION_RATE_LIMIT = 4.9  # queries per second

    intent_trainable = True
    entity_trainable = True

    def __init__(
        self,
        authoring_key=None,
        subscription_key=None,
        app_id=None,
        endpoint=None,
        version="0.1",
    ):
        """Interface for Microsoft LUIS."""
        self._alias = self.name
        endpoint = endpoint or os.getenv("LUIS_ENDPOINT")
        if endpoint is None:
            ValueError(
                "endpoint not passed and not found under environment variable"
                "LUIS_ENDPOINT"
            )
        authoring_key = authoring_key or os.getenv("LUIS_AUTHORING_KEY")
        if authoring_key is None:
            ValueError(
                "authoring_key not passed and not found under environment "
                "variable LUIS_AUTHORING_KEY"
            )
        subscription_key = subscription_key or os.getenv("LUIS_SUBSCRIPTION_KEY")
        if subscription_key is None:
            ValueError(
                "subscription_key not passed and not found under environment "
                "variable LUIS_SUBSCRIPTION_KEY"
            )
        app_id = app_id or os.getenv("LUIS_APP_ID")
        if app_id is None:
            ValueError(
                "app_id not passed and not found under environment variable "
                "LUIS_APP_ID"
            )
        self._endpoint = endpoint
        self._subscription_key = subscription_key
        self._app_id = app_id
        self._version = version
        self._session = requests.Session()
        self._session.headers.update({"Ocp-Apim-Subscription-Key": authoring_key})
        logger.debug(f"Created new app with id {self._app_id}")

    @property  # type: ignore
    @rate_limited(AUTHORING_RATE_LIMIT)
    def requests(self):
        """Util method to access self.session."""
        return self._session

    def _get_base_url(self, cmd, add_version=True):
        base_url = requests.compat.urljoin(self._endpoint, self._app_id)
        base_url += "/"
        if add_version:
            base_url += f"versions/{self._version}/"
        base_url += f"{cmd}"
        return base_url

    def _convert(self, dataset):
        """
        Convert dataset to LUIS format.

        This function accepts a dataset as input and returns a list of samples
        in the vendor specific expected format.
        """
        examples = []

        for text, intent, entities in dataset:
            example = {
                "text": text,
                "intentName": intent,
                "entityLabels": [
                    {
                        "entityName": entity["entity"],
                        "startCharIndex": entity["start"],
                        "endCharIndex": entity["end"],
                    }
                    for entity in entities
                ],
            }  # yapf: disable
            examples.append(example)

        return json.dumps(examples)

    def _add_application(self, name=None, description=None):
        """Create a new LUIS app."""
        timestamp = "{:%Y-%m-%d_%H:%M:%S}".format(datetime.datetime.now())
        body = {
            "name": name or f"test_app_{timestamp}",
            "description": description or "",
            "culture": "de-de",
            "usageScenario": "",
            "domain": "",
            "initialVersionId": self._version,
        }

        response = self.requests.post(self._endpoint, data=body)
        app_id = response.json()
        return app_id

    def _delete_application(self, app_id=None):
        """
        Delete an application.

        If an `app_id` is given that app will be deleted.
        Otherwise `self.app_id
        """
        if app_id is None:
            app_id = self._app_id

        url = requests.compat.urljoin(self._endpoint, app_id)
        response = self.requests.delete(url)
        return response.json()

    def _clone_version(self, old_version="0.1", new_version="perf_test"):
        """
        Clone a version of the application.

        The version '0.1' is kept empty and used as a blank state.
        Then cloned versions are used for the tests.
        """
        self._version = old_version
        url = self._get_base_url("clone", add_version=True)
        body = {"version": new_version}
        response = self.requests.post(url, data=body)
        response = response.json()

        if response == new_version:
            # clone created successfully
            self._version = new_version
            logger.debug(f"Created new version {response}")

        return response

    def _delete_app_version(self, version="perf_test"):
        """Delete an application version."""
        if version is None:
            version = self._version

        if version == "0.1":
            raise Exception("You cant delete the main version")
        self._version = "perf_test"
        url = self._get_base_url("", add_version=True)

        return self._session.delete(url)

    def _create_intents(self, dataset):
        """Add intent classifiers to the application."""
        if dataset.n_intents > 500:
            raise ValueError(
                "LUIS does not support more than 500 intents:\n"
                "https://docs.microsoft.com/en-us/azure/cognitive-services/luis/luis-boundaries"  # noqa: E501
            )

        url = self._get_base_url("intents")
        for intent in dataset.unique_intents:
            body = {"name": intent}
            response = self.requests.post(url, data=body)
            response = response.json()
            logger.debug(f"{response}")

    def _create_entities(self, dataset):
        """Add entity extractors to the application."""
        if dataset.n_entities > 30:
            raise ValueError(
                "LUIS does not support more than 30 entities.\n"
                "https://docs.microsoft.com/en-us/azure/cognitive-services/luis/luis-boundaries"  # noqa: E501
            )

        url = self._get_base_url("entities")
        for entity in dataset.unique_entities:
            body = {"name": entity}
            response = self.requests.post(url, data=body)
            response = response.json()
            logger.debug(f"{response}")

    def _upload_samples(self, dataset):
        logger.debug(f"Uploading samples for dataset {dataset.name}")
        logger.debug(f"Number of samples {dataset.n_samples}")

        logger.debug("Creating intents ...")
        self._create_intents(dataset)
        logger.debug("Creating entities ...")
        self._create_entities(dataset)

        url = self._get_base_url("examples")
        logger.debug("Uploading examples ...")

        # the /examples endpoint only accepts 100 length batches
        step = 100
        for start in range(0, dataset.n_samples, step):
            end = start + step
            idx = slice(start, end)
            ds = dataset[idx]
            body = self._convert(ds)
            response = self.requests.post(url, data=body)
            n_uploaded = len(response.json())
            logger.debug(f"Batch with {n_uploaded} samples uploaded")

        return response.json()

    def _train(self):
        url = self._get_base_url("train", add_version=True)
        logger.debug("Launching training ...")
        response = self.requests.post(url)

        while not self._is_trained:
            logger.debug("Waiting for the model to finish training")
            time.sleep(0.5)
        response = self._publish(staging=True)
        logger.debug("Finished publishing with response %s", response.json())
        return response

    def _get_train_status(self):
        url = self._get_base_url("train", add_version=True)
        response = self.requests.get(url)
        return response.json()

    @property
    def _is_trained(self):
        status = self._get_train_status()
        # logger.debug(f'Training status debug details: {status}')
        status_details = [
            each.get("details", {}).get("status", "") == "Success" for each in status
        ]
        logger.debug(f"Training status details: {status_details}")
        return all(status_details)  # yapf: disable

    def _is_published(self):
        url = self._get_base_url("", add_version=False)
        response = self.requests.get(url)
        logger.debug("Checked publishing with response %s", response.json())
        return response.json()

    def _publish(self, staging=True):
        """Publish a specific version of the application."""
        url = self._get_base_url("publish", add_version=False)
        body = {"versionId": self._version, "isStaging": staging}
        response = self.requests.post(url, data=body)
        return response

    def train_intent(self, dataset):
        """Train intent classifier."""
        logger.info(f"Training on {dataset.n_samples} samples")
        self._delete_app_version(version="perf_test")
        self._clone_version(old_version="0.1", new_version="perf_test")
        self._upload_samples(dataset)

        # not clear if removing "response = "could have side effects
        response = self._train()  # noqa: F841

        logger.debug("Will now start measuring time and calls...")
        global n_calls
        global start_time
        n_calls = 0
        start_time = time.time()
        return self

    @rate_limited(SUBSCRIPTION_RATE_LIMIT)
    def test_single_intent(self, query, return_probs=False):
        """
        Predict the intent for an utterance.

        LUIS doesn't provide a batch testing API so
        we have to test one intent at a time.
        """
        url = self._get_base_url(cmd="", add_version=False)
        url = url.replace("/api", "")
        # utterances have a limit of max 500 characters
        # https://docs.microsoft.com/en-us/azure/cognitive-services/luis/luis-boundaries
        query = query[:500]
        body = {
            "q": query,
            "staging": "true",
            "log": False,
            "subscription-key": self._subscription_key,
        }
        # here we dont use the session because the auth key headers
        # take preference over the subscription key
        response = requests.get(url, params=body)

        # logger.debug(f'Full url: {response.request.url}')

        response = response.json()
        intent = response.get("topScoringIntent", {}).get("intent")
        prob = response.get("topScoringIntent", {}).get("score", 0)

        if intent is None:
            intent = OUT_OF_SCOPE_TOKEN

        global n_calls
        global start_time

        n_calls += 1
        if n_calls % 50 == 0:
            logger.debug(f"Total calls: {n_calls}")
            calls_per_second = n_calls / (time.time() - start_time + 0.0001)
            logger.debug(f"Calls per second: {calls_per_second}")

        if return_probs:
            return intent, prob
        return intent

    def test_intent(self, dataset, return_probs=False):
        """Test intent classifier."""
        logger.info(f"Testing on {dataset.n_samples} samples")
        selfs = [self] * dataset.n_samples
        _return_probs = [return_probs] * dataset.n_samples
        tups = list(zip(selfs, dataset.texts, _return_probs))

        with concurrent.futures.ThreadPoolExecutor(max_workers=15) as executor:
            intents = list(executor.map(_unwrap_self, tups))

        if return_probs:
            intents, probs = list(zip(*intents))
            return intents, probs
        return intents
