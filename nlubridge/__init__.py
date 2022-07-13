# Copyright (c) 2021 Klaus-Peter Engelbrecht, Deutsche Telekom AG
# Copyright (c) 2021 Yaser Martinez-Palenzuela, Deutsche Telekom AG
# This software is distributed under the terms of the MIT license
# which is available at https://opensource.org/licenses/MIT
"""Provides a unified API to several popular intent recognition applications."""  # noqa D400

import sys
from typing import TYPE_CHECKING
# from packaging import version

from lazy_imports import LazyImporter

from .datasets import NLUdataset, OUT_OF_SCOPE_TOKEN, EntityKeys  # noqa: F401


__version__ = "0.2.1dev0"


if TYPE_CHECKING:
    from .dataloaders.luis import from_luis  # noqa: F401
    from .dataloaders.watson import from_watson  # noqa: F401
    from .dataloaders.rasa import from_rasa  # noqa: F401
    from .dataloaders.rasa3 import from_rasa  # noqa: F401
    from .dataloaders.huggingface import from_huggingface  # noqa: F401
    from .dataloaders.utils import from_json, from_csv  # noqa: F401

else:
    _import_structure = {
        "dataloaders.luis": ["from_luis"],
        "dataloaders.watson": ["from_watson"],
        "dataloaders.huggingface": ["from_huggingface"],
        "dataloaders.utils": ["from_json", "from_csv"],
    }
    try:
        from rasa import __version__ as rasa_version
    except ModuleNotFoundError:
        # adding the import to _import_structure gives us an error message related to
        # dependencies being missing rather than not being able to import from here.
        _import_structure.update({"dataloaders.rasa": ["from_rasa"]})
    else:
        # although we know now that Rasa is installed, we still use LazyImporter to
        # import here in case we will have other rasa-specific dependencies in the
        # future
        # if version.parse(rasa_version) < version.parse("3.0.0"):
        if int(rasa_version.split(".")[0]) < 3:
            _import_structure.update({"dataloaders.rasa": ["from_rasa"]})
        else:
            _import_structure.update({"dataloaders.rasa3": ["from_rasa"]})

    sys.modules[__name__] = LazyImporter(
        __name__,
        globals()["__file__"],
        _import_structure,
        # extra_objects={"__version__": __version__},
    )
