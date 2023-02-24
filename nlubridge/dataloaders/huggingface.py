# Copyright (c) 2021 Klaus-Peter Engelbrecht, Deutsche Telekom AG
# This software is distributed under the terms of the MIT license
# which is available at https://opensource.org/licenses/MIT
import datasets

from nlubridge.nlu_dataset import NluDataset


def from_huggingface(
    hugging_ds,
    has_intents=True,
    has_entities=False,
    toks_col="tokens",
    iob_col="ner_tags",
) -> NluDataset:
    """
    Convert a Huggingface dataset to an NLUdataset.

    Requirements if has_entities is True:
        * hugging_ds uses IOB entity annotation format, with tokens
          specified under tok_col (default: 'tokens') and IOB tags
          specified under column iob_col (default: 'ner_tags')
        * hugging_ds provides a ClassLabel object for the IOB tags
          accessible through hugging_ds.features[iob_col].feature

    Requirements if has_intents is True:
        * hugging_ds stores intent information under column 'label'
        * hugging_ds provides a ClassLabel object for the intents
          accessible through hugging_ds.features['label']
        * if has_entities is False, the text to classify needs to be
          stored under column 'text'

    Compatible datasets (examples):
        * Intent: "banking77", "nlu_evaluation_data",
          "snips_built_in_intents"
        * Entities: "wnut_17", "germaner"

    :param hugging_ds: Huggingface dataset from which to import the
        data
    :type hugging_ds: datasets.arrow_dataset.Dataset
    :param has_intents: If True, will try to retrieve an intent label
        for each case
    :type has_intents: bool
    :param has_entities: If True, will try to retrieve entities from an
        attribute that keeps this information in IOB format
    :type has_entities: bool
    :param toks_col: hugging_ds attribute under which the tokenized
        document is found
    :type toks_col: str
    :param iob_col: hugging_ds attribute under which the IOB labels are
        found
    :type iob_col: str
    """
    if isinstance(hugging_ds, datasets.dataset_dict.DatasetDict):
        raise TypeError("Invalid type for argument hugging_ds. Try picking a split.")
    if not isinstance(hugging_ds, datasets.arrow_dataset.Dataset):
        raise TypeError("Invalid type for argument hugging_ds.")

    if has_intents:
        # get intents for each example
        try:
            label_mapper = hugging_ds.features["label"]
        except KeyError:
            raise ValueError(
                "hugging_ds must provide intent classes in a ClassLabel feature named "
                "'label'"
            )
        intents = [label_mapper.int2str(x) for x in hugging_ds["label"]]
    else:
        intents = None  # type: ignore[assignment]

    if has_entities:
        # get entities and texts generated from token lists for each example
        try:
            token_seqs = hugging_ds[toks_col]
        except KeyError:
            raise ValueError(
                "Expect entities in IOB format, which requires a column with tokenized "
                "text. If this is not stored under column 'tokens', provide the column "
                "name as tok_col."
            )
        try:
            ner_tag_seqs = hugging_ds[iob_col]
        except KeyError:
            raise ValueError(
                "Expect entities in IOB format, which requires a column with IOB tags."
                "If this is not stored under column 'ner_tags', provide the column "
                "name as iob_col."
            )
        try:
            tag_mapper = hugging_ds.features[iob_col].feature
        except KeyError:
            raise ValueError(
                "hugging_ds must provide entity tags in a ClassLabel accessible "
                "through the '{}' feature".format(iob_col)
            )
        texts, entities = iob2dict_batch(token_seqs, ner_tag_seqs, tag_mapper)
    else:
        entities = None

    if not has_entities:
        # get text for each example if not yet generated from tokens
        try:
            texts = hugging_ds["text"]
        except KeyError:
            raise ValueError("hugging_ds must have a 'text' feature")

    ds = NluDataset(texts, intents, entities)

    return ds


def iob2dict_batch(token_seqs, iob_seqs, tag_mapper):
    """Run iob2dict for many docs."""
    all_texts = []
    all_entities = []
    for tokens, ner_tags in zip(token_seqs, iob_seqs):
        text, entities = iob2dict(tokens, ner_tags, tag_mapper)
        all_texts.append(text)
        all_entities.append(entities)
    return all_texts, all_entities


def iob2dict(tokens, ner_tags, tag_mapper):
    """
    Convert an IOB labeled doc to dictionary format.

    :param tokens: List of tokens in the document
    :type tokens: List[str]
    :param ner_tags: List of IOB tags
    :type ner_tags: List[str]
    :param tag_mapper: object holding the IOB tag details. Has method
        int2str() which converts the tag to a readable string like
        `B-Person`.
    """

    def need_blank(token, len_text):
        if (
            token.startswith("'")
            or (token in [":", ".", ",", ";", "?", "!", "..."])
            or (len_text == 0)
        ):
            return False
        else:
            return True

    entities = []
    text = ""
    is_in = False

    for token, tag in zip(tokens, ner_tags):
        # remove trailing blank if certain characters follow
        if need_blank(token, len(text)):
            text += " "

        # init or update entity based on tag prefix
        tag = tag_mapper.int2str(tag)
        if tag.startswith("B-"):
            entity = {"start": len(text), "entity": tag.replace("B-", "")}
            is_in = True
        elif tag.startswith("I-"):
            pass  # nothing to do
        elif (tag == "O") and is_in:
            entity["end"] = len(text) - need_blank(token, len(text))
            entities.append(entity)
            is_in = False
        elif tag == "O":
            pass  # nothing to do

        # append current token to text
        text += token

    # make sure entities at the end of an utterance are completed
    if not tag == "O":
        entity["end"] = len(text)
        entities.append(entity)

    return text, entities
