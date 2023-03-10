# Copyright (c) 2021 Klaus-Peter Engelbrecht, Deutsche Telekom AG
# Copyright (c) 2021 Yaser Martinez-Palenzuela, Deutsche Telekom AG
# This software is distributed under the terms of the MIT license
# which is available at https://opensource.org/licenses/MIT

import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

import requests
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
from ibm_watson import AssistantV1
from tqdm import tqdm

from nlubridge.nlu_dataset import OUT_OF_SCOPE_TOKEN

from .vendor import Vendor


logger = logging.getLogger(__name__)


class WatsonAssistant(Vendor):
    def __init__(
        self,
        api_key=None,
        endpoint=None,
        workspace_name=None,
        api_version="2018-02-16",
        max_workers=10,
        use_bulk=True,
    ):
        """
        Interface for IBM Watson Assistant.

        See `IBM Watson Assistant
        <https://www.ibm.com/watson/services/conversation/>`_.

        Uses either of two methods for predictions:
            * use_bulk=True -> uses bulk_classify method of the WA API.
              This is free of charge but available only in premium
              subscriptions. It is also rate limited, possibly making
              it slower in some cases where dataset is long.
              Rate limit parameters are hard-coded, i.e. we need to
              change them in the code if the API changes
            * use_bulk=False -> Makes concurrent requests to the
              message WA API method. For large numbers of test cases,
              this can become expensive.

        :param api_key: see Watson Assistant v1 API docs
        :type api_key: str
        :param endpoint:see Watson Assistant v1 API docs
        :type endpoint: str
        :param workspace_name: see Watson Assistant v1 API docs
        :type workspace_name: str
        :param api_version: see Watson Assistant v1 API docs
        :type api_version: str
        :param max_workers: number of ThreadPoolExecutor workers when use_bulk=False
            (default=10)
        :type max_workers: int
        :param use_bulk: if True (default) uses bulk_classify method
        :type use_bulk: bool
        """
        self._alias = self.name
        api_key = api_key or os.getenv("WATSON_API_KEY")
        if api_key is None:
            ValueError(
                "api_key not passed and not found under environment variable"
                "WATSON_API_KEY"
            )
        endpoint = endpoint or os.getenv("WATSON_ENDPOINT")
        if endpoint is None:
            ValueError(
                "endpoint not passed and not found under environment variable"
                "WATSON_ENDPOINT"
            )
        workspace_name = workspace_name or os.getenv("WATSON_WORKSPACE_NAME")
        if workspace_name is None:
            ValueError(
                "workspace_name not passed and not found under environment "
                "variable WATSON_WORKSPACE_NAME"
            )
        self._assistant = self._connect(endpoint, api_key, api_version)
        self._workspace_name = workspace_name
        self._workspace_id = self._get_workspace_id()
        self._max_workers = max_workers
        self._use_bulk = use_bulk

    def set_bulk(self, use_bulk):
        """
        Set use_bulk on (True) or off (False).

        :param use_bulk: True means use bulk method for testing
        :type use_bulk: bool
        """
        self._use_bulk = use_bulk

    def set_max_workers(self, max_workers):
        """
        Set max_workers for parallel API requests.

        Applicable for processing of test requests if use_bulk=False.

        :param max_workers: max_workers to use for parallel requests
        :type max_workers: int
        """
        self._max_workers = max_workers

    @property
    def _is_trained(self):
        response = self._assistant.get_workspace(
            workspace_id=self._workspace_id, export=False
        ).get_result()
        return response["status"] == "Available"

    def train_intent(self, dataset):
        """Train intent classifier."""
        logger.info(f"Training on {dataset.n_samples} samples")
        self._upload_samples(dataset)
        while not self._is_trained:
            time.sleep(3)
        # Return vendor for compatibility with other vendors implementations
        return self

    def test_intent(
        self,
        dataset,
        return_probs=False,
        max_workers=None,
        n_best_intents=1,
        use_bulk=None,
    ):
        """
        Test intent classifier.

        :param dataset: nlutests dataset
        :type dataset: nlutests.datasets.NLUdataset
        :param return_probs: if True, returns probability of predicted intent
            (default=False)
        :type return_probs: bool
        :param n_best_intents: number of n-best results to return (default=1)
        :type n_best_intents: int
        :param max_workers: deprecated; pass to constructor instead
        :param use_bulk: deprecated; pass to constructor instead
        :return: predictions, probabilities
        """
        if (use_bulk is not None) or (max_workers is not None):
            logger.warning(
                "Deprecated: Please do not pass use_bulk and "
                "max_workers to test_intent anymore but to the "
                "constructor instead. Settings will not apply "
                "if passed to test_intent!"
            )

        logger.info(f"Testing on {dataset.n_samples} samples")

        intents = []
        probs = []
        if not self._is_trained:
            raise RuntimeError("Watson Assistant workspace has not yet been trained")

        if not self._use_bulk:
            responses = self._get_wa_response_in_session(dataset)
        else:
            responses = self._get_wa_response_in_batches(dataset)

        for text, result in zip(dataset.texts, responses):
            intent, prob = self._extract_top_intent_from_result(result, n_best_intents)
            intents.append(intent)
            probs.append(prob)
        if return_probs:
            return intents, probs
        return intents

    @staticmethod
    def validate_text(text):
        """
        Validate an utterance text.

        Validate the test data text so it conforms to the
        Watson Assistant API format, see
        https://www.ibm.com/watson/developercloud/conversation/api/v1/?python#send_message

        :param text: User input
        :type text: str
        """
        val_text = text
        not_allowed = ["\n", "\t", "\r"]
        if len(val_text) > 2048:
            val_text = val_text[0:2047]
        if any(x in val_text for x in not_allowed):
            for char in not_allowed:
                val_text = val_text.replace(char, " ")
        return val_text

    def _connect(self, endpoint, api_key, api_version):
        authenticator = IAMAuthenticator(api_key)
        assistant = AssistantV1(version=api_version, authenticator=authenticator)
        assistant.set_service_url(endpoint)
        return assistant

    def _convert(self, dataset):
        """
        Convert data from the standardized format to Watson Assistant format.

        This function accepts a dataset as input and returns a list of
        samples in the vendor specific expected format.
        https://www.ibm.com/watson/developercloud/conversation/api/v1/?python#create_example

        Texts are modified with self.validate_text to conform to WA
        requirements. In case of duplicates (case-insensitive), only
        the first encountered sample is kept.
        """
        intents_examples = {}
        exist_texts = []
        for text, intent in zip(dataset.texts, dataset.intents):
            text = self.validate_text(text)
            if text.lower() in exist_texts:
                # it's a duplicate text
                continue
            else:
                exist_texts.append(text.lower())
                try:
                    intents_examples[intent].append({"text": text})
                except KeyError:
                    intents_examples[intent] = [{"text": text}]

        data_wa_format = [
            {"intent": intent, "examples": examples}
            for (intent, examples) in intents_examples.items()
        ]
        return data_wa_format

    def _get_workspace_id(self):
        response = self._assistant.list_workspaces().get_result()
        for workspace in response["workspaces"]:
            if workspace["name"] == self._workspace_name:
                return workspace["workspace_id"]
        raise ValueError("Workspace could not be found.")

    def _delete_workspace(self):
        """Delete a workspace by its name."""
        response = self._assistant.delete_workspace(self._workspace_id)
        status_code = response.get_status_code()
        return status_code == "200"

    def _clear_workspace(self):
        """
        Create a fresh WA workspace with the same name as used before.

        The intents in the workspace need to be deleted before a new
        test uploads new intents. This method deletes the existing
        workspace and creates a new one under the same name
        (workspace_id is updated).
        """
        logger.info(f"Clearing workspace {self._workspace_id}")
        # TODO: handle failures of delete_workspace (and fix unused variable 'success')
        success = self._delete_workspace()  # noqa: F841
        response = self._assistant.create_workspace(
            name=self._workspace_name, description="created via api", language="de"
        ).get_result()
        self._workspace_id = response["workspace_id"]

    def _upload_samples(self, dataset):
        self._clear_workspace()
        intents = self._convert(dataset)
        logger.info("Uploading samples")
        for intent in intents:
            # TODO: handle failures (and fix unused variable 'response')
            response = self._assistant.create_intent(  # noqa: F841
                workspace_id=self._workspace_id,
                intent=intent["intent"],
                examples=intent["examples"],
            ).get_result()

    def _get_wa_response(self, query):
        """Get the NLU result for a single message."""
        query = self.validate_text(query)
        response = self._assistant.message(
            workspace_id=self._workspace_id,
            input={"text": query},
            alternate_intents=True,
        )
        return response.get_result()

    def _get_wa_response_in_batches(self, dataset):
        """Get intent predictions using the bulk_classify method."""
        logger.info("Using /bulk_classify endpoint for testing")

        def get_response(queries):
            data = {"input": [{"text": self.validate_text(query)} for query in queries]}
            status_code = 429
            while status_code == 429:
                response = session.post(url, data=json.dumps(data))
                status_code = response.status_code
                if status_code == 429:
                    reset_time = datetime.fromtimestamp(
                        int(response.headers.get("X-RateLimit-Reset", None))
                    )
                    now = datetime.now()
                    wait_seconds = (reset_time - now).total_seconds()
                    logger.info(f"waiting {wait_seconds} seconds (rate limit was hit)")
                    time.sleep(wait_seconds)
            return response

        with requests.Session() as session:
            session.auth = requests.auth.HTTPBasicAuth(
                "apikey", self._assistant.authenticator.token_manager.apikey
            )
            session.headers.update({"Content-Type": "application/json"})
            url = "{url}/v1/workspaces/{ws_id}/bulk_classify?version={version}".format(
                url=self._assistant.service_url,
                ws_id=self._workspace_id,
                version=self._assistant.version,
            )
            all_results = []
            for i in range(0, len(dataset.texts), 50):
                batch = dataset.texts[i : i + 50]
                response = get_response(batch)
                results = response.json()["output"]
                # ensure that dataset.texts and results are correctly aligned
                assert batch == [r["input"]["text"] for r in results]
                all_results += results
        return all_results

    def _get_wa_response_in_session(self, dataset):
        """Get intent predictions with parallelized request."""
        logger.info("Using /message endpoint for testing. Costs may apply!")

        def get_response(query):
            data = {"input": {"text": query}, "alternate_intents": True}
            response = session.post(url, data=json.dumps(data))
            return response

        with requests.Session() as session:
            session.auth = requests.auth.HTTPBasicAuth(
                "apikey", self._assistant.authenticator.token_manager.apikey
            )
            session.headers.update({"Content-Type": "application/json"})
            url = "{url}/v1/workspaces/{ws_id}/message?version={version}".format(
                url=self._assistant.service_url,
                ws_id=self._workspace_id,
                version=self._assistant.version,
            )
            with ThreadPoolExecutor(max_workers=self._max_workers) as executor:
                responses = list(
                    tqdm(
                        executor.map(get_response, dataset.texts),
                        total=len(dataset.texts),
                    )
                )
            all_responses = []
            for response in responses:
                try:
                    r = response.json()
                except requests.exceptions.JSONDecodeError:
                    r = {"error": response.text}
                all_responses.append(r)
            return all_responses

    @staticmethod
    def _extract_top_intent_from_result(wa_result, n_best_intents):
        result_intents = wa_result.get("intents")
        if result_intents:
            if n_best_intents == 1:
                intent = result_intents[0].get("intent")
                prob = result_intents[0].get("confidence")
            else:
                intent = []
                prob = []
                n_best_max = (
                    len(result_intents)
                    if n_best_intents >= len(result_intents)
                    else n_best_intents
                )
                for k in range(0, n_best_max):
                    intent.append(result_intents[k].get("intent"))
                    prob.append(result_intents[k].get("confidence"))
        elif wa_result.get("error"):
            intent = wa_result.get("error")
            prob = -1
        else:
            intent = OUT_OF_SCOPE_TOKEN
            prob = 0
        return intent, prob
