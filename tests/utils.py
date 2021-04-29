import os

import transformers

import class_attention as cat
from tests.test_training_utils import DATASET

GLOVE_TMP_PATH = "glove.glove"
EXTRA_CLASSES_TMP_PATH = "extra_classes.txt"
CLASS_NAMES = {
    "ARTS",
    "ARTS & CULTURE",
    "BLACK VOICES",
    "BUSINESS",
    "COLLEGE",
    "COMEDY",
    "CRIME",
    "CULTURE & ARTS",
    "DIVORCE",
    "EDUCATION",
    "ENTERTAINMENT",
    "ENVIRONMENT",
    "FIFTY",
    "FOOD & DRINK",
    "GOOD NEWS",
    "GREEN",
    "HEALTHY LIVING",
    "HOME & LIVING",
    "IMPACT",
    "LATINO VOICES",
    "MEDIA",
    "MONEY",
    "PARENTING",
    "PARENTS",
    "POLITICS",
    "QUEER VOICES",
    "RELIGION",
    "SCIENCE",
    "SPORTS",
    "STYLE",
    "STYLE & BEAUTY",
    "TASTE",
    "TECH",
    "THE WORLDPOST",
    "TRAVEL",
    "WEDDINGS",
    "WEIRD NEWS",
    "WELLNESS",
    "WOMEN",
    "WORLD NEWS",
    "WORLDPOST",
}


def make_glove_file():
    # if os.path.exists(GLOVE_TMP_PATH):
    #     raise RuntimeError("glove_tmp_path exists")

    with open(GLOVE_TMP_PATH, "w") as f:
        used_names = set()
        for class_name in CLASS_NAMES:
            for word in class_name.split(" "):
                word = word.lower()
                if word in used_names:
                    continue

                f.write(f"{word} 42 42 42\n")
                used_names.add(word)


def delete_glove_file():
    os.remove(GLOVE_TMP_PATH)


def make_extra_classes_file():
    with open(EXTRA_CLASSES_TMP_PATH, "w") as f:
        for class_name in CLASS_NAMES:
            f.write(class_name + "\n")


def delete_extra_classes_file():
    os.remove(EXTRA_CLASSES_TMP_PATH)


def model_factory(txt_encoder_kwargs=None, cls_encoder_kwargs=None, model_kwargs=None):
    txt_encoder_kwargs = txt_encoder_kwargs or dict()
    cls_encoder_kwargs = cls_encoder_kwargs or dict()
    model_kwargs = model_kwargs or dict()

    text_encoder = transformers.AutoModel.from_config(
        transformers.BertConfig(
            hidden_size=32,
            num_hidden_layers=2,
            intermediate_size=64,
            num_attention_heads=4,
            hidden_dropout_prob=0,
            attention_probs_dropout_prob=0,
            **txt_encoder_kwargs,
        )
    )
    label_encoder = transformers.AutoModel.from_config(
        transformers.BertConfig(
            hidden_size=32,
            num_hidden_layers=2,
            intermediate_size=64,
            num_attention_heads=4,
            hidden_dropout_prob=0,
            attention_probs_dropout_prob=0,
            **cls_encoder_kwargs,
        )
    )

    model = cat.ClassAttentionModel(text_encoder, label_encoder, hidden_size=7, **model_kwargs)
    return model


def default_prepare_dataloaders(**kwargs):
    _kwargs = dict(
        dataset_name_or_path=DATASET,
        test_class_frac=0.2,
        batch_size=8,
        model_name="distilbert-base-uncased",
        dataset_frac=0.005,
        num_workers=0,
        text_field="headline",
        class_field="category",
        test_set_name="validation",
    )

    for key, value in kwargs.items():
        _kwargs[key] = value

    return cat.training_utils.prepare_dataloaders(
        **_kwargs,
    )
