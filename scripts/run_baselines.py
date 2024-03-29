import argparse
import logging
import os
import sys
import json

import torch
import torch.utils.data
import transformers

import numpy as np
import sklearn
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer

import class_attention as cat


logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    stream=sys.stdout,
)
logger = logging.getLogger(os.path.basename(__file__))
logging.getLogger("transformers.configuration_utils").setLevel(logging.WARNING)
logging.getLogger("wandb.sdk.internal.internal").setLevel(logging.WARNING)

MODEL = "distilbert-base-uncased"


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)

    return {"accuracy": np.sum(predictions == labels) / predictions.shape[0]}


def compute_metrics_full(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)

    # we do this to unnumpy tensors into lists
    labels_str = [str(l) for l in labels]
    predictions_str = [str(p) for p in predictions]

    metrics, _ = cat.evaluation_utils.compute_metrics(y_true=labels_str, y_pred=predictions_str)
    metrics["accuracy"] = metrics["acc"]

    return metrics


def parse_args(args=None):
    parser = argparse.ArgumentParser()

    # fmt: off
    # data
    parser.add_argument("--dataset",
                        help="Name or path to a HuggingFace Datasets dataset")
    # architecture
    parser.add_argument("--bert-model", default=MODEL, type=str)

    # training
    parser.add_argument("--max-epochs", default=10, type=int)
    parser.add_argument("--batch-size", default=32, type=int)
    parser.add_argument("--lr", default=1e-4, type=float)

    # misc
    parser.add_argument("--device", default=None)
    # fmt: on

    args = parser.parse_args(args)
    args.device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    return args


def main(args):
    logger.info(f"Starting the script with the arguments \n{json.dumps(vars(args), indent=4)}")
    logger.info("Preparing the data")
    text_field, class_field = cat.utils.infer_field_names(dataset_name=args.dataset)

    news_dataset = cat.utils.get_dataset_by_name_or_path(args.dataset)

    train_set = news_dataset["train"]
    valid_set = news_dataset["validation"]

    train_classes = list(set(train_set[class_field]))
    class2id = {c: i for i, c in enumerate(train_classes)}

    valid_set = valid_set.filter(lambda x: x[class_field] in train_classes)

    logger.info("Training TF-IDF vectorizer")

    vectorizer = TfidfVectorizer()
    X_train = vectorizer.fit_transform(train_set[text_field])
    y_train = train_set[class_field]

    logger.info("Training Linear SVM classifier")
    model = LinearSVC()
    model.fit(X_train, y_train)

    X_test = vectorizer.transform(valid_set[text_field])
    y_test = valid_set[class_field]

    predictions = model.predict(X_test)

    linear_model_accuracy = sklearn.metrics.accuracy_score(y_pred=predictions, y_true=y_test)
    linear_model_f1_score = sklearn.metrics.f1_score(
        y_pred=predictions, y_true=y_test, average="macro"
    )
    linear_model_f1_score_weighted = sklearn.metrics.f1_score(
        y_pred=predictions, y_true=y_test, average="weighted"
    )

    logger.info(f"Linear model test accuracy   : {linear_model_accuracy}")
    logger.info(f"Linear model test F1 macro   : {linear_model_f1_score}")
    logger.info(f"Linear model test F1 weighted: {linear_model_f1_score_weighted}")

    logger.info("Loading BERT model")
    logger.info(f"Model name: {args.bert_model}")

    tokenizer = transformers.AutoTokenizer.from_pretrained(args.bert_model, use_fast=True)
    model = transformers.AutoModelForSequenceClassification.from_pretrained(
        args.bert_model, num_labels=len(train_classes)
    )

    def preprocess_function(examples):
        return {
            **tokenizer(examples[text_field], truncation=True),
            "label": [class2id[c] for c in examples[class_field]],
        }

    encoded_train_set = train_set.map(preprocess_function, batched=True)
    encoded_valid_set = valid_set.map(preprocess_function, batched=True)

    training_args = transformers.TrainingArguments(
        output_dir="debug_outputs",
        evaluation_strategy=transformers.EvaluationStrategy.EPOCH,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.max_epochs,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
    )

    trainer = transformers.Trainer(
        model,
        training_args,
        train_dataset=encoded_train_set,
        eval_dataset=encoded_valid_set,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics_full,
    )

    trainer.train()

    # trainer.compute_metrics = compute_metrics_full
    # prediction_output_obj = trainer.predict(test_dataset=encoded_valid_set)
    # wandb.log(prediction_output_obj.metrics)

    logger.info("Script finished successfully")


if __name__ == "__main__":
    args = parse_args()
    main(args)
