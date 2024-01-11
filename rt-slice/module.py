import os
import pandas as pd
import inspect
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple
from collections import Counter
import logging

from terminator.tokenization import ExpressionBertTokenizer
from sklearn.utils import shuffle
from transformers.training_args import TrainingArguments

from transformers import (
    LineByLineTextDataset,
)


import tempfile
import shutil
from transformers import (
    CONFIG_MAPPING,
    AutoConfig,
    AutoModelWithLMHead,
    set_seed,
)
import json


from transformers import (
    DataCollatorForPermutationLanguageModeling,
)
from terminator.collators import TRAIN_COLLATORS


logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


TRANSFORM_FACTORY = {"SELFIES": None, "SLICES": lambda x: x}
AUGMENT_FACTORY = {
    "SMILES": None,
    "SELFIES":None,
    "AAS": None,
}



class Property:
    name: str
    minimum: float = 0
    maximum: float = 0
    expression_separator: str = "|"
    normalize: bool = False

    def __init__(self, name: str):
        self.name = name
        if not name.strip("<>").isalnum():
            raise ValueError(f"Properties have to be alphanumerics, not {name}.")
        self.mask_lengths: List = []

    def update(self, line: str):
        prop = line.split(self.name)[-1].split(self.expression_separator)[0]
        try:
            val = float(prop)
        except ValueError:
            logger.error(f"Could not convert property {prop} in {line} to float.")
        if val < self.minimum:
            self.minimum = val
        elif val > self.maximum:
            self.maximum = val
        self.mask_lengths.append(len(prop))

    @property
    def mask_length(self) -> int:
        """
        How many tokens are being masked for this property.
        """
        counts = Counter(self.mask_lengths)
        if len(counts) > 1:
            logger.warning(
                f"Not all {self.name} properties have same number of tokens: {counts}"
            )
        return int(counts.most_common(1)[0][0])
    
def add_tokens_from_lists(
    tokenizer: ExpressionBertTokenizer, train_data: List[str], test_data: List[str]
) -> Tuple[ExpressionBertTokenizer, Dict[str, Property], List[str], List[str]]:
    """
    Addding tokens to a tokenizer from parsed datasets hold in memory.

    Args:
        tokenizer: The tokenizer.
        train_data: List of strings, one per sample.
        test_data: List of strings, one per sample.

    Returns:
       Tuple with:
            tokenizer with updated vocabulary.
            dictionary of property names and full property objects.
            list of strings with training samples.
            list of strings with testing samples.
    """
    num_tokens = len(tokenizer)
    properties: Dict[str, Property] = {}
    all_tokens: Set = set()
    for data in [train_data, test_data]:
        for line in data:
            # Grow the set of all tokens in the dataset
            toks = tokenizer.tokenize(line)
            all_tokens = all_tokens.union(toks)
            # Grow the set of all properties (assumes that the text follows the last `|`)
            props = [
                x.split(">")[0] + ">"
                for x in line.split(tokenizer.expression_separator)[:-1]
            ]
            for prop in props:
                if prop not in properties.keys():
                    properties[prop] = Property(prop)
                properties[prop].update(line)

    # Finish adding new tokens
    tokenizer.add_tokens(list(all_tokens))
    tokenizer.update_vocab(all_tokens)  # type:ignore
    logger.info(f"Added {len(tokenizer)-num_tokens} new tokens to tokenizer.")

    return tokenizer, properties, train_data, test_data

def prepare_datasets_from_files(
    tokenizer: ExpressionBertTokenizer,
    train_path: str,
    test_path: str,
    augment: int = 0,
) -> Tuple[ExpressionBertTokenizer, Dict[str, Property], List[str], List[str]]:
    """
    Converts datasets saved in provided `.csv` paths into RT-compatible datasets.
    NOTE: Also adds the new tokens from train/test data to provided tokenizer.

    Args:
        tokenizer: The tokenizer.
        train_path: Path to the training data.
        test_path: Path to the testing data.
        augment: Factor by which each training sample is augmented.

    Returns:
       Tuple with:
            tokenizer with updated vocabulary.
            dict of property names and property objects.
            list of strings with training samples.
            list of strings with testing samples.
    """

    # Setup data transforms and augmentations
    train_data: List[str] = []
    test_data: List[str] = []
    properties: List[str] = []

    aug = AUGMENT_FACTORY.get(tokenizer.language, lambda x: x)
    trans = TRANSFORM_FACTORY.get(tokenizer.language, lambda x: x)

    for i, (data, path) in enumerate(
        zip([train_data, test_data], [train_path, test_path])
    ):

        if not path.endswith(".csv"):
            raise TypeError(f"Please provide a csv file not {path}.")

        # Load data
        df = shuffle(pd.read_csv(path))
        if "text" not in df.columns:
            raise ValueError("Please provide text in the `text` column.")

        if i == 1 and set(df.columns) != set(properties + ["text"]):
            raise ValueError(
                "Train and test data have to have identical columns, not "
                f"{set(properties + ['text'])} and {set(df.columns)}."
            )
        properties = sorted(list(set(properties).union(list(df.columns))))
        properties.remove("text")

        # Parse data and create RT-compatible format
        for j, row in df.iterrows():
            line = "".join(
                [
                    f"<{p}>{row[p]:.3f}{tokenizer.expression_separator}"
                    for p in properties
                ]
            #    + [trans(row.text)]  # type: ignore
            #    + [trans(row.text)]

            )
            data.append(line)

        # Perform augmentation on training data if applicable
        if i == 0 and augment is not None and augment > 1:
            for _ in range(augment):
                for j, row in df.iterrows():
                    line = "".join(
                        [
                            f"<{p}>{row[p]:.3f}{tokenizer.expression_separator}"
                            for p in properties
                        ]
                    #    + [trans(aug(row.text))]  # type: ignore
                    )
                    data.append(line)

    return add_tokens_from_lists(
        tokenizer=tokenizer, train_data=train_data, test_data=test_data
    )



def get_train_config_dict(
    training_args: Dict[str, Any], properties: Set
) -> Dict[str, Any]:
    return {
        "alternate_steps": training_args["alternate_steps"],
        "reset_training_loss": True,
        "cg_collator": training_args["cg_collator"],
        "cc_loss": training_args["cc_loss"],
        "property_tokens": list(properties),
        "cg_collator_params": {
            "do_sample": False,
            "property_tokens": list(properties),
            "plm_probability": training_args["plm_probability"],
            "max_span_length": training_args["max_span_length"],
            "entity_separator_token": training_args["entity_separator_token"],
            "mask_entity_separator": training_args["mask_entity_separator"],
            "entity_to_mask": training_args["entity_to_mask"],
        },
    }


def get_hf_training_arg_object(training_args: Dict[str, Any]) -> TrainingArguments:
    """
    A method to convert a training_args Dictionary into a HuggingFace
    `TrainingArguments` object.
    This routine also takes care of removing arguments that are not necessary.

    Args:
        training_args: A dictionary of training arguments.

    Returns:
        object of type `TrainingArguments`.
    """

    # Get attributes of parent class
    org_attrs = dict(inspect.getmembers(TrainingArguments))

    # Remove attributes that were specified by child classes
    hf_training_args = {k: v for k, v in training_args.items() if k in org_attrs.keys()}

    # Instantiate class object
    hf_train_object = TrainingArguments(training_args["output_dir"])

    # Set attributes manually (since this is a `dataclass` not everything can be passed
    # to constructor)
    for k, v in hf_training_args.items():
        setattr(hf_train_object, k, v)

    return hf_train_object