# Copyright (c) 2021 Yaser Martinez-Palenzuela, Deutsche Telekom AG
# This software is distributed under the terms of the MIT license
# which is available at https://opensource.org/licenses/MIT

import collections
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional, Union

from nlubridge.nlu_dataset import EntityKeys, NluDataset


def from_csv(
    path: Union[Path, str], text_col: Union[str, int], intent_col: Union[str, int]
) -> NluDataset:
    """
    Load dataset (only text and intents) from a csv file.

    text_col and intent_col can be either integer indices of the respective columns
    (start index is 0), or, if the first row of the file holds the column names, the
    repective column name strings.
    """
    columns = collections.defaultdict(list)

    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            for k, v in row.items():
                columns[k].append(v)

    if isinstance(text_col, int):
        key = list(columns.keys())[text_col]
        texts = [key, *columns[key]]
    else:
        texts = columns[text_col]

    if isinstance(intent_col, int):
        key = list(columns.keys())[intent_col]
        intents = [key, *columns[key]]
    else:
        intents = columns[intent_col]

    ds = NluDataset(texts, intents)
    return ds


def _convert_entities_for_nludataset(
    entities, type_key, start_key, end_key, end_index_add_1: bool
):
    ex_entities = []
    for entity in entities:
        formatted_entity = {
            EntityKeys.TYPE: entity[type_key],
            EntityKeys.START: entity[start_key],
            EntityKeys.END: entity[end_key] + end_index_add_1,
        }
        # Add any custom keys defined in the source json
        for key in entity.keys():
            if key not in [type_key, start_key, end_key]:
                formatted_entity[key] = entity[key]
        ex_entities.append(formatted_entity)
    return ex_entities


def from_json(
    path: Optional[Union[Path, str]] = None,
    examples: Optional[List[Dict]] = None,
    text_key: str = "text",
    intent_key: str = "intent",
    entities_key: str = "entities",
    entity_type_key: str = "entity",
    entity_start_key: str = "start",
    entity_end_key: str = "end",
    end_index_add_1: bool = False,
):
    """
    Load the dataset form a json string.

    The json string should have the structure in the TestDataset class.
    """
    if (not path and not examples) or (path and examples):
        raise ValueError("Exactly one of path or examples arguments needs to be given")

    if path:
        with open(path, "r") as f:
            examples = json.load(f)

    texts, intents, entities = [], [], []
    for example in examples:  # type: ignore[union-attr]
        texts.append(example[text_key])
        intents.append(example[intent_key])
        example_entities = _convert_entities_for_nludataset(
            example[entities_key],
            entity_type_key,
            entity_start_key,
            entity_end_key,
            end_index_add_1,
        )
        entities.append(example_entities)
    dataset = NluDataset(texts, intents, entities)
    return dataset
