import argparse
import logging
import pprint
import os
import sys
from itertools import chain

import torch
import torch.utils.data
import torch.nn.functional as F
import transformers
import wandb

from tqdm import tqdm

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


def parse_args(args=None):
    parser = argparse.ArgumentParser()

    # fmt: off
    parser.add_argument("--test-class-frac", default=0.2, type=float,
                        help="a fraction of classes to remove from the training set (and use for zero-shot)")
    parser.add_argument("--dataset-frac", default=1.0, type=float,
                        help="a fraction of dataset to train and evaluate on, used for debugging")

    parser.add_argument("--hidden-size", default=128, type=int)

    parser.add_argument("--max-epochs", default=10, type=int)
    parser.add_argument("--batch-size", default=32, type=int)
    parser.add_argument("--lr", default=1e-4, type=float)

    parser.add_argument("--device", default=None)
    # fmt: on

    args = parser.parse_args(args)
    args.device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    return args


def prepare_dataloaders(text_class_frac, dataset_frac):
    reduced_train_set, test_set, all_classes_str, test_classes_str = cat.training_utils.prepare_dataset(
        text_class_frac, dataset_frac
    )

    text_tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL, fast=True)
    label_tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL, fast=True)

    # Datasets
    reduced_train_dataset = cat.CatDataset(
        reduced_train_set["headline"],
        text_tokenizer,
        reduced_train_set["category"],
        label_tokenizer,
    )
    test_dataset = cat.CatDataset(
        reduced_train_set["headline"],
        text_tokenizer,
        reduced_train_set["category"],
        label_tokenizer,
    )

    # Dataloaders
    all_classes_ids = label_tokenizer.batch_encode_plus(
        all_classes_str, return_tensors="pt", add_special_tokens=True, padding=True,
    )["input_ids"]

    train_collator = cat.CatCollator(pad_token_id=label_tokenizer.pad_token_id)
    test_collator = cat.CatTestCollator(
        possible_labels_ids=all_classes_ids, pad_token_id=label_tokenizer.pad_token_id
    )

    train_dataloader = torch.utils.data.DataLoader(
        reduced_train_dataset, batch_size=args.batch_size, collate_fn=train_collator
    )
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, collate_fn=test_collator
    )
    return train_dataloader, test_dataloader, all_classes_str, test_classes_str


def main(args):
    logger.info("Creating dataloaders")
    train_dataloader, test_dataloader, all_classes_str, test_classes_str = prepare_dataloaders(
        args.test_class_frac, args.dataset_frac
    )

    # Model
    logger.info("Creating model and optimizer")
    text_encoder = transformers.AutoModel.from_pretrained(MODEL)
    label_encoder = transformers.AutoModel.from_pretrained(MODEL)

    model = cat.ClassAttentionModel(text_encoder, label_encoder, hidden_size=args.hidden_size)
    model = model.to(args.device)

    parameters = chain(
        model.txt_encoder.parameters(), model.txt_out.parameters(), model.cls_out.parameters()
    )
    optimizer = torch.optim.Adam(parameters, lr=args.lr)

    logger.info("Starting training")
    wandb.init(
        project="class_attention",
        tags=["notebooks"],
        notes=" ",
        config={"test_classes": ",".join(test_classes_str)},
    )
    wandb.watch(model)

    for _ in tqdm(range(args.max_epochs)):
        for x, c, y in train_dataloader:
            optimizer.zero_grad()

            x = x.to(args.device)
            c = c.to(args.device)
            y = y.to(args.device)

            x_dict = {"input_ids": x}
            c_dict = {"input_ids": c}
            logits = model(x_dict, c_dict)

            loss = F.cross_entropy(logits, y)

            _, preds = logits.max(-1)
            acc = torch.sum(preds == y).float() / x.shape[0]

            # fmt: off
            wandb.log({
                "train/acc": acc,
                "train/loss": loss,
            })
            # fmt: on

            loss.backward()
            optimizer.step()

        # validation
        metrics = cat.utils.evaluate_model_per_class(
            model,
            test_dataloader,
            device=args.device,
            labels_str=all_classes_str,
            zeroshot_labels=test_classes_str,
        )
        wandb.log({f"eval/{k}": v for k, v in metrics.items()})

    logger.info("Script finished successfully")


if __name__ == "__main__":
    args = parse_args()
    main(args)
