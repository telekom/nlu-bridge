"""Implementations of interfaces to different vendors."""
import sys
from typing import TYPE_CHECKING

from lazy_imports import LazyImporter

from .vendors import Vendor  # noqa: 401

if TYPE_CHECKING:

    from .tfidf_intent_classifier import TfidfIntentClassifier  # noqa: 401
    from .luis import LUIS  # noqa: F401
    from .watson import Watson  # noqa: F401
    from .spacy import SpacyClassifier  # noqa: F401
    from .telekom import TelekomModel  # noqa: F401
    from .fasttext import FastText  # noqa: F401
    from .rasa import Rasa  # noqa: F401
    from .rasa3 import Rasa3  # noqa: F401
else:
    _import_structure = {
        "tfidf_intent_classifier": ["TfidfIntentClassifier"],
        "luis": ["LUIS"],
        "watson": ["Watson"],
        "spacy": ["SpacyClassifier"],
        "telekom": ["TelekomModel"],
        "fasttext": ["FastText"],
        "rasa": ["Rasa"],
        "rasa3": ["Rasa3"],
    }
    sys.modules[__name__] = LazyImporter(
        __name__,
        globals()["__file__"],
        _import_structure,
        # extra_objects={"__version__": __version__},
    )
