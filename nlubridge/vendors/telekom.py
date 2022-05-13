# Copyright (c) Deutsche Telekom AG
# This software is distributed under the terms of the MIT license
# which is available at https://opensource.org/licenses/MIT

import itertools
import pickle
import logging

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.svm import OneClassSVM
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from fuzzywuzzy import fuzz
import numpy as np

from .vendors import Vendor
from ..datasets import OUT_OF_SCOPE_TOKEN


logger = logging.getLogger(__name__)


class TelekomModel(Vendor):
    alias = "telekom_model"

    def __init__(self, anomaly_clf_nu=0):
        """
        Interface to a Telekom custom intent classifier.

        Intent classification is based on character n-grams TFIDF
        model with hyper parameter tuning. In addition, a second
        classifier detects out-of-scope examples with an anomaly
        detection approach.

        :param anomaly_clf_nu: parameter that sets the
            aggressiveness of the anomaly detection (set to 0 to switch
            off)
        :type anomaly_clf_nu: float
        """
        self.clf = Model1(
            none_class=OUT_OF_SCOPE_TOKEN, verbose=False, anomaly_clf_nu=anomaly_clf_nu
        )

    def train_intent(self, dataset):  # noqa D102
        X = dataset.texts
        y = dataset.intents
        self.clf.fit(X, y)
        return self

    def test_intent(self, dataset, return_probs=False):  # noqa D102
        X = dataset.texts
        intents, probs = self.clf.predict(X, return_probs=True)
        intents = list(intents)
        probs = list(probs)
        if return_probs:
            return intents, probs
        return intents


class TelekomModel2(Vendor):
    alias = "telekom_model_2"

    def __init__(self):
        """Alternative model custom-built for Telekom."""
        self.clf = Model2(none_class=OUT_OF_SCOPE_TOKEN, verbose=True)

    def train_intent(self, dataset):  # noqa D102
        X = dataset.texts
        y = dataset.intents
        self.clf.fit(X, y)

    def test_intent(self, dataset):  # noqa D102
        X = dataset.texts
        return self.clf.predict(X)


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


class Model2:
    """
    Model2 implementation in original code.

    TF-IDF ==> Fuzzy string matching
    """

    def __init__(self, none_class="no_intent", verbose=False):  # noqa D102

        self.none_class_ = none_class
        self.verbose_ = verbose
        self.filename = "models/pickles/model1.pkl"

        self.tfidf_vectorizer = TfidfVectorizer(
            min_df=2, max_df=0.9, ngram_range=(2, 5), use_idf=True, analyzer="word"
        )

    def fit(self, X, y=None):  # noqa D102
        logger.debug("Training started")

        self.classes_ = list(set(y) - set([self.none_class_]))

        self.key_words = {}

        for c in self.classes_:
            if not c == self.none_class_:
                X_c = [x for x, y_c in zip(X, y) if c == y_c]

                self.tfidf_vectorizer.fit(X_c)
                response = self.tfidf_vectorizer.transform(X_c)

                feature_array = np.array(self.tfidf_vectorizer.get_feature_names())
                tfidf_sorting = np.argsort(response.toarray()).flatten()[::-1]

                n = 10
                top_n = feature_array[tfidf_sorting][:n]

                self.key_words[c] = top_n

        # self.save()

        logger.debug("Training finished")

    def predict(self, X, y=None):  # noqa D102
        logger.debug("Prediction started")
        y_pred = []

        for i, x in enumerate(X):
            matches = {}
            for c in self.classes_:
                predict = self._get_n_gram_matches(x, self.key_words[c])
                if len(predict) > 0:
                    matches[c] = predict[0]["distance"]
            if len(matches) > 0:
                y_pred.append(max(matches, key=matches.get))
            else:
                y_pred.append(self.none_class_)

        logger.debug("Prediction finished")
        return y_pred

    def load(self):  # noqa D102
        f = open(self.filename, "rb")
        tmp_dict = pickle.load(f)
        f.close()

        self.__dict__.update(tmp_dict)

    def save(self):  # noqa D102
        f = open(self.filename, "wb")
        pickle.dump(self.__dict__, f, 2)
        f.close()

    def _get_n_gram_matches(self, inputString, options, isReordered=False):  # noqa D102
        max_n_gram = 4
        ngrams = []

        splitInputString = inputString.split()
        for i in range(3, max_n_gram + 1):
            ngrams.extend(self._create_n_gram(splitInputString, i, isReordered))

        matches = []
        for i in ngrams:
            matches.extend(self._norm_best_match_substring(i, options))

        return matches

    def _create_n_gram(self, inputWords, n, reorder=False):  # noqa D102
        ngrams = []
        for i in range(0, len(inputWords) + 1 - n):
            strings = inputWords[i : i + n]
            if reorder:
                permutations = list(itertools.permutations(strings))
                for p in permutations:
                    ngrams.append(" ".join(p))
            else:
                ngrams.append(" ".join(strings))
        return ngrams

    def _norm_best_match_substring(self, item, options, cutoff=0.75):  # noqa D102
        tuples = []
        for i in options:
            tuples.append(
                {
                    "string": item,
                    "value": i,
                    "distance": (
                        (fuzz.token_sort_ratio(i.lower(), item.lower())) / 100.0
                    ),
                }
            )
        sortedOptions = sorted(tuples, key=lambda x: x["distance"], reverse=True)

        filteredOptions = [x for x in sortedOptions if x["distance"] > cutoff]
        return filteredOptions
