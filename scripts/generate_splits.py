"""
Produce splits for the zero-shot version of Fraser/news-category-dataset dataset.
Makes 4 datasets: semantic, semantic-small, random, and random-small.
For the semantic split, the test classes are pre-selected for this split based on their names.
For the random split, they were selected randomly.

This script is generated from notebooks/12_static_split.ipynb
It expects a directory named "data" in the .. directory
"""
import random
import datasets
import numpy as np

import class_attention as cat

random.seed(57)
np.random.seed(57)

# Semantic Split
# The test classes are pre-selected for this split based on their names

news_dataset = datasets.load_dataset("Fraser/news-category-dataset")

# ### A note on labels
#
# The dataset we use (`Fraser/news-category-dataset`) has some interesting particularities in the class names.
#
# For example, it has classes `STYLE` and `STYLE & BEAUTY` or `WORLD NEWS` and `NEWS`.
# I.e., some classes contain other classes names in their name.
# The classes that have `&` in their name have a similar particularity.
# Some of the categories does not seem to be distinguishable.
# E.g., `THE WORLDPOST` and `WORLDPOST` or `ARTS & CULTURE` and `CULTURE & ARTS`.
#
# * &	: STYLE & BEAUTY, ARTS & CULTURE, HOME & LIVING, FOOD & DRINK, CULTURE & ARTS
# * VOICES	: LATINO VOICES, BLACK VOICES, QUEER VOICES
# * NEWS	: WEIRD NEWS, GOOD NEWS, WORLD NEWS
# * ARTS	: ARTS, ARTS & CULTURE, CULTURE & ARTS
# * CULTURE	: ARTS & CULTURE, CULTURE & ARTS
# * LIVING	: HEALTHY LIVING, HOME & LIVING
# * WORLDPOST	: THE WORLDPOST, WORLDPOST
# * WORLD	: THE WORLDPOST, WORLDPOST

test_classes = [
    "LATINO VOICES",  # related to BLACK VOICES, QUEER VOICES,
    "PARENTS",  # related to PARENTING
    "DIVORCE",  # related to WEDDINGS
    "WORLDPOST",  # related to THE WORLDPOST,
    "STYLE",  # related to STYLE & BEAUTY,
    "WORLD NEWS",  # related to WEIRD NEWS, GOOD NEWS
    "CULTURE & ARTS",  # related to ARTS & CULTURE
    "EDUCATION",  # related to COLLEGE, SCIENCE,
]

reduced_train_set, _ = cat.utils.split_classes(
    news_dataset["train"], test_classes=test_classes, verbose=True
)

news_dataset["train"] = reduced_train_set
news_dataset.save_to_disk("../data/news-category-semantic-split")

p = 0.1
news_dataset["train"] = cat.utils.sample_dataset(news_dataset["train"], p)
news_dataset["validation"] = cat.utils.sample_dataset(news_dataset["validation"], p)
news_dataset["test"] = cat.utils.sample_dataset(news_dataset["test"], p)

news_dataset.save_to_disk("../data/news-category-semantic-split-small")


# Random split
# The test classes are randomly chosen

news_dataset = datasets.load_dataset("Fraser/news-category-dataset")

test_classes_random = [
    "COLLEGE",
    "ARTS",
    "POLITICS",
    "RELIGION",
    "WORLD NEWS",
    "SPORTS",
    "EDUCATION",
    "TRAVEL",
]
reduced_train_set, _ = cat.utils.split_classes(
    news_dataset["train"],
    test_classes=test_classes_random,
    verbose=True,
)
news_dataset["train"] = reduced_train_set

news_dataset.save_to_disk("../data/news-category-random-split")

p = 0.1
news_dataset["train"] = cat.utils.sample_dataset(news_dataset["train"], p)
news_dataset["validation"] = cat.utils.sample_dataset(news_dataset["validation"], p)
news_dataset["test"] = cat.utils.sample_dataset(news_dataset["test"], p)

news_dataset.save_to_disk("../data/news-category-random-split-small")

# repeat this to have an extremely small dataset (1/1000 laof the original) used for debugging

p = 0.01
news_dataset["train"] = cat.utils.sample_dataset(news_dataset["train"], p)
news_dataset["validation"] = cat.utils.sample_dataset(news_dataset["validation"], p)
news_dataset["test"] = cat.utils.sample_dataset(news_dataset["test"], p)

news_dataset.save_to_disk("../data/news-category-random-split-tiny")

print("Script finished successfully")
