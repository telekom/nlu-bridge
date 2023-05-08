# Copyright (c) 2021 Klaus-Peter Engelbrecht, Deutsche Telekom AG
# Copyright (c) 2021 Yaser Martinez-Palenzuela, Deutsche Telekom AG
# This software is distributed under the terms of the MIT license
# which is available at https://opensource.org/licenses/MIT
"""Provides a unified API to several popular intent recognition applications."""  # noqa D400

import sys
from typing import TYPE_CHECKING

from lazy_imports import LazyImporter


__version__ = "1.0.1"


if TYPE_CHECKING:
    from .dataloaders.huggingface import from_huggingface  # noqa: F401
    from .dataloaders.luis import from_luis  # noqa: F401
    from .dataloaders.rasa import from_rasa, to_rasa  # noqa: F401, F811
    from .dataloaders.utils import from_csv, from_json  # noqa: F401
    from .dataloaders.watson_assistant import from_watson_assistant  # noqa: F401
    from .nlu_dataset import (  # noqa: F401
        OUT_OF_SCOPE_TOKEN,
        EntityKeys,
        NBestKeys,
        NluDataset,
        concat,
    )
    from .vendors.vendor import Vendor  # noqa: 401

else:
    _import_structure = {
        "dataloaders.huggingface": ["from_huggingface"],
        "dataloaders.luis": ["from_luis"],
        "dataloaders.rasa": ["from_rasa", "to_rasa"],
        "dataloaders.utils": ["from_json", "from_csv"],
        "dataloaders.watson_assistant": ["from_watson_assistant"],
        "nlu_dataset": [
            "OUT_OF_SCOPE_TOKEN",
            "EntityKeys",
            "NBestKeys",
            "NluDataset",
            "concat",
        ],
        "vendors.vendor": ["Vendor"],
    }

    sys.modules[__name__] = LazyImporter(
        __name__,
        globals()["__file__"],
        _import_structure,
        # extra_objects={"__version__": __version__},
    )
