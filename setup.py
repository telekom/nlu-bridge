# Copyright (c) 2021 Klaus-Peter Engelbrecht, Deutsche Telekom AG
# This software is distributed under the terms of the MIT license
# which is available at https://opensource.org/licenses/MIT

"""Build script for setuptools."""

import os

from setuptools import setup, find_packages


project_name = "nlubridge"
source_code = "https://github.com/telekom/nlu-bridge"
keywords = [
    "nlu",
    "intent recognition",
    "natural language understanding",
    "evaluation",
    "performance",
]


def get_version():
    """Read version from ``__init__.py``."""
    version_filepath = os.path.join(
        os.path.dirname(__file__), project_name, "__init__.py"
    )
    with open(version_filepath) as f:
        for line in f:
            if line.startswith("__version__"):
                return line.strip().split()[-1][1:-1]
    assert False


with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()


setup(
    name=project_name,
    version=get_version(),
    maintainer="Klaus-Peter Engelbrecht",
    author="Klaus-Peter Engelbrecht",
    author_email="k.engelbrecht@telekom.de",
    description=(
        "Provides a unified API to several popular intent recognition applications"
    ),
    long_description=long_description,
    long_description_content_type="text/markdown",
    keywords=keywords,
    url=source_code,
    project_urls={
        "Bug Tracker": source_code + "/issues",
        # "Documentation": "https://telekom.github.io/HPOflow/",
        "Source Code": source_code,
        "Contributing": source_code + "/blob/main/CONTRIBUTING.md",
        "Code of Conduct": source_code + "/blob/main/CODE_OF_CONDUCT.md",
    },
    packages=find_packages(),
    python_requires=">=3, <3.9",
    install_requires=["sklearn", "python-dotenv", "lazy-imports", "ratelimit"],
    extras_require={
        "watson": ["ibm_watson", "tqdm", "requests"],
        "telekom": ["fuzzywuzzy", "python-Levenshtein"],
        "fasttext": ["fasttext"],
        "luis": ["requests", "ratelimit"],
        "rasa": ["rasa==2"],
        "rasa3": ["rasa==3"],
        "spacy": ["spacy==3.1.3"],
        "develop": [
            "pytest-cov",
            "pytest-mock",
            "flake8",
            "black",
            "pydocstyle",
            "setuptools",
            "wheel",
            "twine",
        ],
    },
    include_package_data=True,
)
