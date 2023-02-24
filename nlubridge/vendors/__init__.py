# Copyright (c) 2021 Klaus-Peter Engelbrecht, Deutsche Telekom AG
# This software is distributed under the terms of the MIT license
# which is available at https://opensource.org/licenses/MIT
"""Implementations of interfaces to different vendors."""

import sys
from typing import TYPE_CHECKING

from lazy_imports import LazyImporter

from .vendor import Vendor  # noqa: 401


if TYPE_CHECKING:
    from .char_ngram_intent_classifier import CharNgramIntentClassifier  # noqa: F401
    from .fasttext import FastText  # noqa: F401
    from .luis import Luis  # noqa: F401
    from .rasa2 import Rasa2  # noqa: F401
    from .rasa3 import Rasa3  # noqa: F401
    from .spacy import Spacy  # noqa: F401
    from .tfidf_intent_classifier import TfidfIntentClassifier  # noqa: 401
    from .vendor import Vendor  # noqa: 401
    from .watson_assistant import WatsonAssistant  # noqa: F401
else:
    _import_structure = {
        "fasttext": ["FastText"],
        "luis": ["Luis"],
        "rasa2": ["Rasa2"],
        "rasa3": ["Rasa3"],
        "spacy": ["Spacy"],
        "char_ngram_intent_classifier": ["CharNgramIntentClassifier"],
        "tfidf_intent_classifier": ["TfidfIntentClassifier"],
        "vendor": ["Vendor"],
        "watson_assistant": ["WatsonAssistant"],
    }
    sys.modules[__name__] = LazyImporter(
        __name__,
        globals()["__file__"],
        _import_structure,
        # extra_objects={"__version__": __version__},
    )
