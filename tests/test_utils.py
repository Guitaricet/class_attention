import pytest

import torch
import torch.utils.data
import transformers
import datasets

import class_attention as cat


torch.manual_seed(93)


@pytest.fixture()
def random_model():
    random_text_encoder = transformers.BertModel(
        transformers.BertConfig(num_hidden_layers=2, intermediate_size=256)
    )
    random_label_encoder = transformers.BertModel(
        transformers.BertConfig(num_hidden_layers=2, intermediate_size=256)
    )
    random_model = cat.ClassAttentionModel(
        random_text_encoder, random_label_encoder, hidden_size=768
    )

    return random_model


@pytest.fixture()
def train_texts():
    # fmt: off
    train_texts = [
        "This is a news article",
        "Good news, everyone!",
        "It is cloudy with a chance of meatballs",
        "This is a sport article",
    ]
    # fmt: on
    return train_texts


@pytest.fixture()
def train_labels():
    return ["News", "News", "Weather", "Sport"]


@pytest.fixture()
def possible_labels_str():
    return ["News", "Weather", "Sport"]


@pytest.fixture()
def arrow_dataset():
    return datasets.load_dataset("Fraser/news-category-dataset")["validation"]


@pytest.fixture()
def dataloader(train_texts, train_labels, possible_labels_str):
    text_tokenizer = cat.utils.make_whitespace_tokenizer(train_texts)
    label_tokenizer = cat.utils.make_whitespace_tokenizer(train_labels, unk_token=None)

    dataset = cat.CatDataset(train_texts, text_tokenizer, train_labels, label_tokenizer)

    possible_labels_ids = torch.LongTensor(
        [label_tokenizer.encode(l).ids for l in possible_labels_str]
    )
    test_collator = cat.CatTestCollator(possible_labels_ids=possible_labels_ids, pad_token_id=0)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, collate_fn=test_collator)
    return dataloader


def test_valiadte_model_per_class_on_dataloader(random_model, dataloader):
    labels = ["My News", "Weather", "Sport"]
    metrics = cat.evaluation_utils.evaluate_model(
        random_model, dataloader, device="cpu", labels_str=labels, zeroshot_labels=["My News"]
    )

    assert 0 <= metrics["eval/acc"] <= 1
    assert 0 <= metrics["eval/P_macro"] <= 1
    assert 0 <= metrics["eval/R_macro"] <= 1
    assert 0 <= metrics["eval/F1_macro"] <= 1

    assert 0 <= metrics["zero_shot_eval/acc"] <= 1
    assert 0 <= metrics["zero_shot_eval/P_macro"] <= 1
    assert 0 <= metrics["zero_shot_eval/R_macro"] <= 1
    assert 0 <= metrics["zero_shot_eval/F1_macro"] <= 1

    assert 0 <= metrics["multi_shot_eval/acc"] <= 1
    assert 0 <= metrics["multi_shot_eval/P_macro"] <= 1
    assert 0 <= metrics["multi_shot_eval/R_macro"] <= 1
    assert 0 <= metrics["multi_shot_eval/F1_macro"] <= 1

    at_least_one = False
    for metric in ["P", "R"]:
        for label in labels:
            m = metrics[f"eval_per_class/{label}/{metric}"]
            assert 0 <= m <= 1
            if m > 0 and m < 1:
                at_least_one = True

    assert at_least_one, "all metrics are either 0 or 1"


def test_split_classes(arrow_dataset):
    test_classes = ["RELIGION", "EDUCATION", "ARTS", "TRAVEL"]
    train_dataset, test_dataset = cat.utils.split_classes(
        arrow_dataset, class_field="category", test_classes=test_classes, verbose=False
    )

    assert set(train_dataset["category"]).isdisjoint(set(test_dataset["category"]))
    assert set(test_dataset["category"]) == set(test_classes)


def test_split_classes2(arrow_dataset):
    test_classes = ["RELIGION", "EDUCATION", "ARTS", "TRAVEL"]
    train_dataset, test_dataset = cat.utils.split_classes(
        arrow_dataset, class_field="category", test_classes=test_classes
    )

    assert set(train_dataset["category"]).isdisjoint(set(test_dataset["category"]))
    assert set(test_dataset["category"]) == set(test_classes)


def test_split_classes_no_zero_shot(arrow_dataset):
    train_dataset, test_dataset = cat.utils.split_classes(
        arrow_dataset, class_field="category", p_test_classes=0, test_classes=None, verbose=False
    )

    assert test_dataset is None


def test_get_cced(random_model):
    tokenizer = transformers.AutoTokenizer.from_pretrained("bert-base-uncased")
    train_classes_str = ["train", "classes", "str"]
    test_classes_str = ["test", "labels"]

    cced_0 = cat.utils.get_cced(
        model=random_model,
        train_classes_str=train_classes_str,
        test_classes_str=train_classes_str,
        label_tokenizer=tokenizer,
        device="cpu",
    )
    assert cced_0.item() == 0

    cced = cat.utils.get_cced(
        model=random_model,
        train_classes_str=train_classes_str,
        test_classes_str=test_classes_str,
        label_tokenizer=tokenizer,
    )
    assert cced.item() > 0


def test_get_rmasp(random_model):
    tokenizer = transformers.AutoTokenizer.from_pretrained("bert-base-uncased")
    train_classes_str = ["train", "classes", "str"]
    test_classes_str = ["test", "labels"]

    rmasp_0 = cat.utils.get_rmasp(
        model=random_model,
        train_classes_str=train_classes_str,
        test_classes_str=train_classes_str,
        label_tokenizer=tokenizer,
        device="cpu",
    )
    assert rmasp_0.item() > 0

    rmasp = cat.utils.get_rmasp(
        model=random_model,
        train_classes_str=train_classes_str,
        test_classes_str=test_classes_str,
        label_tokenizer=tokenizer,
    )
    assert rmasp.item() > 0
    assert rmasp.item() < rmasp_0.item()
