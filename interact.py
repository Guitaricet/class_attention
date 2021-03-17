import argparse
import logging
import os
import sys
import json
from pprint import pformat

import torch
import torch.utils.data
import torch.nn.functional as F
import transformers
import wandb

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


def parse_args(args=None):
    parser = argparse.ArgumentParser()

    # fmt: off
    parser.add_argument("--device", default=None)
    parser.add_argument("--checkpoint-path", default=None, type=str,
                        help="Checkpoint file to load the model state, optimizer state and script arguments")
    # fmt: on

    args = parser.parse_args(args)
    args.device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    return args


def encode_classes(classes_str, label_tokenizer):
    return label_tokenizer.batch_encode_plus(
        classes_str,
        return_tensors="pt",
        add_special_tokens=True,
        padding=True,
    )["input_ids"]


def main(args):
    logger.info(f"Starting the script with the arguments \n{json.dumps(vars(args), indent=4)}")

    logger.info(f"Loading the checkpoint from {args.checkpoint_path}")
    checkpoint = torch.load(args.checkpoint_path)
    saved_args = checkpoint["model_args"]

    logger.info(f"Building the model from args: {saved_args}")
    model = cat.ClassAttentionModel.from_kwargs(**saved_args)

    logger.info("Restoring model state")
    state_dict = checkpoint["model_state_dict"]
    model.load_state_dict(state_dict)
    model.eval()

    logger.info("Loading tokenizers")
    text_tokenizer = transformers.AutoTokenizer.from_pretrained(saved_args["model"], fast=True)

    if saved_args["glove"] is not None:
        _, word2id = cat.utils.load_glove_from_file(saved_args["glove"])
        label_tokenizer = cat.utils.GloVeTokenizer(word2id)
    else:
        label_tokenizer = transformers.AutoTokenizer.from_pretrained(
            saved_args["model"], fast=True
        )

    test_classes_str = checkpoint["train_classes_str"]
    test_classes_ids = encode_classes(test_classes_str, label_tokenizer)

    logger.info(f"The model is ready. The default classes are: {test_classes_str}.")
    logger.info("To replace them use `set_classes: class_name1 class_name2` (no commas between class names)")
    logger.info("To add classes use `add_classes: class_name3 class_name4`")
    logger.info("To exit press Ctrl+C or print `exit()`")

    while True:
        input_str = input("Print text and the model will predict its class:\n")
        if input_str == "exit()":
            break

        # set_classes handler
        if input_str.startswith("set_classes"):
            if "," in input_str:
                logger.warning("Found comma in class names, it will be a part of the class name.")
                logger.warning("Using commas is not recommended, split class names with spaces only")

            new_classes = input_str.split(" ")[1:]
            if len(test_classes_str) < 2:
                logger.warning("You should provide more than two class names. Aborting set_classes command")
                continue

            if not all(len(c) > 0 for c in new_classes):
                logger.warning("One of the classes is an empty string. Aborting set_classes command")
                continue

            test_classes_str = new_classes
            test_classes_ids = encode_classes(test_classes_str, label_tokenizer)
            logger.info(f"Set new class names: {new_classes}")
            del new_classes
            continue

        # add_classes handler
        if input_str.startswith("add_classes"):
            new_classes = input_str.split(" ")[1:]

            if not isinstance(new_classes, list):
                raise ValueError(new_classes)

            if len(new_classes) < 1:
                logger.warning("You should provide at leas one class. Aborting add_classes command")
                continue

            if not all(len(c) > 0 for c in new_classes):
                logger.warning("One of the classes is an empty string. Aborting add_classes command")
                continue

            test_classes_str.extend(new_classes)
            test_classes_ids = encode_classes(test_classes_str, label_tokenizer)
            continue

        # default handler: classify provided text

        text_ids = cat.CatDataset.encode_via_tokenizer(input_str, text_tokenizer)
        assert len(text_ids.shape) == 1, text_ids.shape
        text_ids = text_ids.unsqueeze(0)

        with torch.no_grad():
            logits = model(text_ids, test_classes_ids).squeeze(0)

            probs = torch.softmax(logits, dim=-1)
            _, pred = probs.max(dim=-1)
            pred = pred.item()

            probs = probs.detach().cpu().tolist()
            print(f"Predicted class: {test_classes_str[pred]} with probability {probs[pred]}")

            distribution = {c: round(p, 4) for c, p in zip(test_classes_str, probs)}
            distribution = sorted(distribution.items(), key=lambda x: x[1])
            print(f"Probability distribution: {pformat(distribution)}")

    logger.info("Script finished successfully")


if __name__ == "__main__":
    args = parse_args()
    main(args)
