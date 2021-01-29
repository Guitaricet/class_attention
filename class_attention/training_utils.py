import datasets

import class_attention as cat


def prepare_dataset(test_class_frac, dataset_frac=1.0):
    # TODO: test me
    news_dataset = datasets.load_dataset("Fraser/news-category-dataset")
    train_set = news_dataset["train"]
    test_set = news_dataset["validation"]
    all_classes = list(set(news_dataset["train"]["category"]))

    if dataset_frac < 1:
        # some magic is happening here to make a toy dataset that is consistent, read carefully
        train_set = cat.utils.sample_dataset(news_dataset["train"], p=dataset_frac)

        classes_left = list(set(train_set["category"]))

        test_set = news_dataset["validation"]
        if len(all_classes) > len(classes_left):
            _, test_set = cat.utils.split_classes(test_set)

        test_set = cat.utils.sample_dataset(test_set, p=dataset_frac)

    reduced_train_set, _train_set_remainder = cat.utils.split_classes(
        train_set, p_test_classes=test_class_frac, verbose=True
    )
    test_classes = list(set(_train_set_remainder["category"]))

    return reduced_train_set, test_set, all_classes, test_classes
