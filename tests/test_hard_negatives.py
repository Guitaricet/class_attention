import pytest
import os

import torch
import numpy as np
import transformers

import class_attention as cat
import tests.utils


batch_size = 32
n_examples = 1000
emb_name = "not_text_emb"
text_ids_name = "not_text_ids"
label_ids_name = "not_label_ids"


@pytest.fixture()
def dataset_str():
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        "distilbert-base-uncased", use_fast=True
    )
    dataset_str = tests.utils.dataset_factory(split=f"train[:{n_examples}]")

    def map_fn(items):
        text_ids = tokenizer(items["headline"], return_attention_mask=False)["input_ids"]
        title_ids = tokenizer(items["category"], return_attention_mask=False)["input_ids"]
        emb = np.random.rand(len(items["headline"]), 128)

        return {emb_name: emb, text_ids_name: text_ids, label_ids_name: title_ids}

    dataset_str = dataset_str.map(map_fn, batched=True)

    dataset_str.add_faiss_index(emb_name)
    return dataset_str


def test_hard_sampler_05(dataset_str):
    sampler_05 = cat.hard_negatives.HardNegativeDatasetWAug(
        dataset=dataset_str,
        batch_size=batch_size,
        text_ids_field=text_ids_name,
        label_ids_field=label_ids_name,
        tokenizer=None,
        index_field=emb_name,
        collator=cat.CatCollator(pad_token_id=0),
        percent_hard=0.5,
    )

    x, y, c = next(iter(sampler_05))
    assert x.shape[0] == batch_size


def test_hard_sampler_09(dataset_str):
    sampler_09 = cat.hard_negatives.HardNegativeDatasetWAug(
        dataset=dataset_str,
        batch_size=batch_size,
        text_ids_field=text_ids_name,
        label_ids_field=label_ids_name,
        tokenizer=None,
        index_field=emb_name,
        collator=cat.CatCollator(pad_token_id=0),
        percent_hard=0.9,
    )

    x, y, c = next(iter(sampler_09))
    assert x.shape[0] == batch_size

    # test that average minimum similarity between the first 16 is way higher that between the last 16?

