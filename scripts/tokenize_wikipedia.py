import argparse
import logging
import os
import sys
import json

import datasets
import transformers


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
    parser.add_argument("--dataset",
                        help="Name or path to a HuggingFace Datasets dataset")
    parser.add_argument("--tokenizer", default=MODEL, type=str)
    parser.add_argument("--num-proc", default=20, type=int)
    # fmt: on

    args = parser.parse_args()

    return args


def main(args):
    logger.info(f"Starting the script with the arguments \n{json.dumps(vars(args), indent=4)}")

    logging.info("Loading data")
    dataset_dict = datasets.load_from_disk(args.dataset)

    logging.info("Loading tokenizer")
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)

    def encode_fn(items):
        text_ids = tokenizer(items["text"], return_attention_mask=False)["input_ids"]
        title_ids = tokenizer(items["title"], return_attention_mask=False)["input_ids"]

        categories_ids = []

        for categories in items["categories"]:
            categories_ids.append([tokenizer(c, return_attention_mask=False)["input_ids"] for c in categories])

        return {"text_ids": text_ids, "title_ids": title_ids, "categories_ids": categories_ids}

    logging.info("Tokenizing")
    dataset_dict = dataset_dict.map(encode_fn, batched=True, num_proc=args.num_proc)

    logging.info("Saving to disk")
    dataset_dict.save_to_disk(args.dataset + "_tokenized_" + args.tokenizer)

    logging.info("Script finished successfully")


if __name__ == "__main__":
    args = parse_args()
    main(args)
