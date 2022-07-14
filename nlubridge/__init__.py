# Copyright (c) 2021 Klaus-Peter Engelbrecht, Deutsche Telekom AG
# Copyright (c) 2021 Yaser Martinez-Palenzuela, Deutsche Telekom AG
# This software is distributed under the terms of the MIT license
# which is available at https://opensource.org/licenses/MIT
"""Provides a unified API to several popular intent recognition applications."""  # noqa D400

import sys
from typing import TYPE_CHECKING

from lazy_imports import LazyImporter


__version__ = "0.2.1dev0"


if TYPE_CHECKING:
    from .dataloaders.luis import from_luis  # noqa: F401
    from .dataloaders.watson import from_watson  # noqa: F401
    from .dataloaders.rasa import from_rasa, to_rasa  # noqa: F401, F811
    from .dataloaders.huggingface import from_huggingface  # noqa: F401
    from .dataloaders.utils import from_json, from_csv  # noqa: F401
    from .datasets import NLUdataset, OUT_OF_SCOPE_TOKEN, EntityKeys  # noqa: F401

else:
    _import_structure = {
        "dataloaders.luis": ["from_luis"],
        "dataloaders.watson": ["from_watson"],
        "dataloaders.huggingface": ["from_huggingface"],
        "dataloaders.rasa": ["from_rasa", "to_rasa"],
        "dataloaders.utils": ["from_json", "from_csv"],
        "datasets": ["NLUdataset", "OUT_OF_SCOPE_TOKEN", "EntityKeys"],
    }

    sys.modules[__name__] = LazyImporter(
        __name__,
        globals()["__file__"],
        _import_structure,
        # extra_objects={"__version__": __version__},
    )
