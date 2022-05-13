"""Implementations of interfaces to different vendors."""

from .vendors import Vendor  # noqa: 401

from lazy_imports import try_import

with try_import() as optional_watson_import:
    from .tfidf_intent_classifier import TfidfIntentClassifier  # noqa: F401
    from .luis import LUIS  # noqa: F401
    from .watson import Watson  # noqa: F401
    from .spacy import SpacyClassifier  # noqa: F401
    from .telekom import TelekomModel  # noqa: F401
    from .fasttext import FastText  # noqa: F401

with try_import() as optional_rasa_import:
    from .rasa import Rasa  # noqa: F401

with try_import() as optional_rasa3_import:
    from .rasa3 import Rasa3  # noqa: F401
