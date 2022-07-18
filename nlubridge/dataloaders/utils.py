# Copyright (c) 2021 Yaser Martinez-Palenzuela, Deutsche Telekom AG
# This software is distributed under the terms of the MIT license
# which is available at https://opensource.org/licenses/MIT

import json
import csv
import collections

from nlubridge.datasets import NluDataset, EntityKeys


def from_csv(filepath, text_col, intent_col) -> NluDataset:
    """Load dataset (only text and intents) from a csv file."""
    columns = collections.defaultdict(list)

    with open(filepath) as f:
        reader = csv.DictReader(f)
        for row in reader:
            for (k, v) in row.items():
                columns[k].append(v)

    if isinstance(text_col, int):
        key = list(columns.keys())[text_col]
        columns[text_col] = [key, *columns[key]]

    if isinstance(intent_col, int):
        key = list(columns.keys())[intent_col]
        columns[intent_col] = [key, *columns[key]]

    ds = NluDataset(columns[text_col], columns[intent_col])
    return ds


def from_json(
    json_string,
    text_key="text",
    intent_key="intent",
    entities_key="entities",
    entity_type_key="entity",
    entity_start_key="start",
    entity_end_key="end",
    end_index_add_1=False,
):
    """
    Load the dataset form a json string.

    The json string should have the structure in the TestDataset class.
    """

    def format_entities(entities, type_key, start_key, end_key, end_index_add_1):
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

    examples = json.loads(json_string)
    texts, intents, entities = [], [], []
    for example in examples:
        texts.append(example[text_key])
        intents.append(example[intent_key])
        example_entities = format_entities(
            example[entities_key],
            entity_type_key,
            entity_start_key,
            entity_end_key,
            end_index_add_1,
        )
        entities.append(example_entities)
    dataset = NluDataset(texts, intents, entities)
    return dataset
