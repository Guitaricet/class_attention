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
    # data
    parser.add_argument("--test-class-frac", default=0.2, type=float,
                        help="a fraction of classes to remove from the training set (and use for zero-shot)")
    parser.add_argument("--dataset-frac", default=1.0, type=float,
                        help="a fraction of dataset to train and evaluate on, used for debugging")

    # architecture
    parser.add_argument("--model", default=MODEL, type=str)
    parser.add_argument("--hidden-size", default=128, type=int)
    parser.add_argument("--normalize-txt", default=False, action="store_true")
    parser.add_argument("--normalize-cls", default=False, action="store_true")
    parser.add_argument("--scale-attention", default=False, action="store_true",
                        help="we recommend to use scaling if normalization is not used")
    parser.add_argument("--temperature", default=0., type=float,
                        help="softmax temperature (used as the initial value if --learn-temperature")
    parser.add_argument("--learn-temperature", default=False, action="store_true",
                        help="learn the softmax temperature as an additional scalar parameter")
    parser.add_argument("--remove-n-lowest-pc", default=0, type=int,
                        help="remove n lowest principal components from the class embeddings")
    parser.add_argument("--use-n-projection-layers", default=0, type=int,
                        help="transform text embedding and class embedding using FCN with this many layers; "
                             "nonlinearity is not used if n=1")
    parser.add_argument("--attention-type", default="dot-product",
                        choices=["dot-product", "bahdanau"])

    # training
    parser.add_argument("--max-epochs", default=10, type=int)
    parser.add_argument("--batch-size", default=32, type=int)
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--freeze-projections", default=False, action="store_true",
                        help="do not train cls_out and txt_out")
    parser.add_argument("--freeze-cls-network", default=False, action="store_true")
    parser.add_argument("--share-txt-cls-network-params", default=False, action="store_true")

    # misc
    parser.add_argument("--device", default=None)
    parser.add_argument("--debug", default=False, action="store_true",
                        help="overrides the arguments for a faster run (smaller model, smaller dataset)")
    # fmt: on

    args = parser.parse_args(args)
    args.device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    if args.share_txt_cls_network_params:
        raise NotImplementedError()

    if args.debug:
        args.dataset_frac = 0.01

    return args


def main(args):
    wandb.init(project="class_attention", config=args)

    logger.info("Creating dataloaders")
    (
        train_dataloader,
        test_dataloader,
        all_classes_str,
        test_classes_str,
    ) = cat.training_utils.prepare_dataloaders(args.test_class_frac, args.batch_size, args.model, args.dataset_frac)
    wandb.config.test_classes = ",".join(sorted(test_classes_str))

    # Model
    logger.info("Creating model and optimizer")

    if args.debug:
        text_encoder = cat.modelling_utils.get_small_transformer()
        label_encoder = cat.modelling_utils.get_small_transformer()
    else:
        text_encoder = transformers.AutoModel.from_pretrained(args.model)
        label_encoder = transformers.AutoModel.from_pretrained(args.model)

    model = cat.ClassAttentionModel(
        text_encoder,
        label_encoder,
        **vars(args),
    )
    model = model.to(args.device)

    parameters = model.get_trainable_parameters()
    optimizer = torch.optim.Adam(parameters, lr=args.lr)

    logger.info("Starting training")
    wandb.watch(model, log="all")

    for _ in tqdm(range(args.max_epochs), desc="Epochs"):
        for x, c, y in train_dataloader:
            optimizer.zero_grad()

            x = x.to(args.device)
            c = c.to(args.device)
            y = y.to(args.device)

            x_dict = {"input_ids": x}
            c_dict = {"input_ids": c}
            logits = model(x_dict, c_dict)  # [batch_size, n_classes]

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
