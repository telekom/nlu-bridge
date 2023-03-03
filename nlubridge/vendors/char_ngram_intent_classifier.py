# Copyright (c) Deutsche Telekom AG
# This software is distributed under the terms of the MIT license
# which is available at https://opensource.org/licenses/MIT

import logging
import pickle

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.svm import OneClassSVM

from ..nlu_dataset import OUT_OF_SCOPE_TOKEN
from .vendor import Vendor


logger = logging.getLogger(__name__)


class CharNgramIntentClassifier(Vendor):
    def __init__(self, anomaly_clf_nu=0):
        """
        Interface to a custom intent classifier based on character n-grams.

        Intent classification is based on character n-grams TFIDF
        model with hyper parameter tuning. In addition, a second
        classifier detects out-of-scope examples with an anomaly
        detection approach.

        :param anomaly_clf_nu: parameter that sets the
            aggressiveness of the anomaly detection (set to 0 to switch
            off)
        :type anomaly_clf_nu: float
        """
        self._alias = self.name
        self._clf = Model1(
            none_class=OUT_OF_SCOPE_TOKEN, verbose=False, anomaly_clf_nu=anomaly_clf_nu
        )

    def train_intent(self, dataset):  # noqa D102
        X = dataset.texts
        y = dataset.intents
        self._clf.fit(X, y)
        return self

    def test_intent(self, dataset, return_probs=False):  # noqa D102
        X = dataset.texts
        intents, probs = self._clf.predict(X, return_probs=True)
        intents = list(intents)
        probs = list(probs)
        if return_probs:
            return intents, probs
        return intents


class Model1:
    """
    Model1 implementation in original code.

    TF-IDF ==> Anomaly
    TF-IDF ==> SGD
    """

    def __init__(
        self, none_class="no_intent", verbose=False, anomaly_clf_nu=0.95
    ):  # noqa D102
        self.none_class_ = none_class
        self.verbose_ = verbose
        self.anomaly_clf_nu_ = anomaly_clf_nu
        self.models_path = "models/"
        self.anomaly_clf = Pipeline(
            [
                ("tfidf", TfidfVectorizer(max_df=1.0, min_df=2, analyzer="word")),
                ("svm", OneClassSVM(kernel="linear", nu=self.anomaly_clf_nu_)),
            ]
        )
        self.intent_clf = Pipeline(
            [
                (
                    "tfidf",
                    TfidfVectorizer(
                        min_df=1,
                        max_df=1.0,
                        ngram_range=(1, 4),
                        use_idf=True,
                        analyzer="char",
                    ),
                ),
                ("sgd", SGDClassifier(alpha=7e-06, max_iter=20, n_jobs=1, loss="log")),
            ]
        )

        self.intent_clf_param_space = {
            "tfidf__max_df": [0.5, 0.75, 1.0],
            "tfidf__ngram_range": [(1, 4), (1, 5), (1, 6)],
            "sgd__alpha": [1e-05, 1e-06, 1e-07],
        }

    def fit(self, X, y=None):  # noqa D102
        logger.debug("Training started")
        X_with_intent = [x for x, y in zip(X, y) if not y == self.none_class_]
        if self.anomaly_clf_nu_ > 0:
            self.trained_anomaly_clf = self.anomaly_clf.fit(X_with_intent)
        self.trained_intent_clf = self._hyper_fit(
            self.intent_clf, self.intent_clf_param_space, X, y
        )
        # self.save()
        logger.debug("Training finished")

    def predict(self, X, y=None, return_probs=False):  # noqa D102
        if self.anomaly_clf_nu_ > 0:
            y_has_intent = self.trained_anomaly_clf.predict(X)

        y_intent = self.trained_intent_clf.predict(X)
        probs = self.trained_intent_clf.predict_proba(X)
        probs = probs.max(axis=1)

        if self.anomaly_clf_nu_ > 0:
            intents = [
                y if has_intent < 0 else self.none_class_
                for y, has_intent in zip(y_intent, y_has_intent)
            ]
        else:
            intents = y_intent

        if return_probs:
            return intents, probs

        return intents

    def _hyper_fit(self, classifier, param_space, X, y):  # noqa D102
        search = GridSearchCV(
            classifier, param_space, n_jobs=-1, cv=5, verbose=self.verbose_, refit=True
        )
        # t = time()
        search.fit(X, y)

        # TODO: This code snippet prints some diagnostic info re the hyperparameter
        #  search. We may want to reactivate it, but then maybe output to logger?
        # if self.verbose_:
        #     pprint(param_space)
        #     print("grid search time for ", time() - t, "s")
        #     print("Best score: %0.3f" % search.best_score_)
        #     print("Best parameters set:")
        #     best_parameters = search.best_params_
        #     for param_name in sorted(param_space.keys()):
        #         print("\t%s: %r" % (param_name, best_parameters[param_name]))

        return search

    def load(self, filename="model1"):  # noqa D102
        f = open(self.models_path + filename, "rb")
        tmp_dict = pickle.load(f)
        f.close()

        self.__dict__.update(tmp_dict)

    def save(self, filename="model1"):  # noqa D102
        f = open(self.models_path + filename, "wb")
        pickle.dump(self.__dict__, f, 2)
        f.close()
