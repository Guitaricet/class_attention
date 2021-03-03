import argparse
import logging
import os
import sys
import json

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

MODEL = "distilbert-base-uncased"
DATASET = "Fraser/news-category-dataset"


def parse_args(args=None):
    parser = argparse.ArgumentParser()

    # fmt: off
    # data
    parser.add_argument("--dataset", default=DATASET,
                        help="Name or path to a HuggingFace Datasets dataset")
    parser.add_argument("--test-class-frac", default=0.0, type=float,
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
    parser.add_argument("--use-n-projection-layers", default=None, type=int,
                        help="transform text embedding and class embedding using FCN with this many layers; "
                             "nonlinearity is not used if n=1")
    parser.add_argument("--attention-type", default="dot-product",
                        choices=["dot-product", "bahdanau"])
    parser.add_argument("--bahdanau-layers", default=1, type=int,
                        help="number of layers in the bahdanau attention network")

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
    parser.add_argument("--predict-into-folder", default=None, type=str,
                        help="Specify this to save predictions into a bunch of files in this folder.")
    parser.add_argument("--tags", default=None)

    args = parser.parse_args(args)
    args.device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    args.tags = args.tags.split(",") if args.tags else []

    # fmt: on

    if args.share_txt_cls_network_params:
        raise NotImplementedError()

    if args.debug:
        logger.info(
            "Running in a debug mode, overriding dataset, tags, max_epochs, dataset_frac, and test_class_frac args"
        )
        args.dataset = DATASET
        args.dataset_frac = 0.01
        args.test_class_frac = 0.2
        args.max_epochs = 2
        args.tags = ["debug"]

    return args


def main(args):
    wandb.init(project="class_attention", config=args, tags=args.tags)
    logger.info(f"Starting the script with the arguments \n{json.dumps(vars(args), indent=4)}")

    logger.info("Creating dataloaders")
    (
        train_dataloader,
        test_dataloader,
        all_classes_str,
        test_classes_str,
        data,
    ) = cat.training_utils.prepare_dataloaders(
        dataset_name_or_path=args.dataset,
        model_name=args.model,
        test_class_frac=args.test_class_frac,
        dataset_frac=args.dataset_frac,
        batch_size=args.batch_size,
    )
    wandb.config.test_classes = ",".join(sorted(test_classes_str))

    cat.training_utils.validate_dataloader(test_dataloader, test_classes_str, is_test=True)
    cat.training_utils.validate_dataloader(train_dataloader, test_classes_str, is_test=False)

    logger.info(f"List of zero-shot classes: {', '.join(test_classes_str)}")

    if len(test_classes_str) < 2:
        logger.warning(f"Less than two zero-shot classes")

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
    wandb.log({"model_description": wandb.Html(cat.utils.monospace_html(repr(model)))})

    logger.info("Starting training")

    predict_into_file = None
    if args.predict_into_folder is not None:
        os.makedirs(args.predict_into_folder, exist_ok=True)
        predict_into_file = os.path.join(args.predict_into_folder, "predictions_all_classes.txt")

    cat.training_utils.train_cat_model(
        model=model,
        optimizer=optimizer,
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        all_classes_str=all_classes_str,
        test_classes_str=test_classes_str,
        max_epochs=args.max_epochs,
        device=args.device,
        predict_into_file=predict_into_file,
    )

    if args.predict_into_folder is not None:
        wandb.save(predict_into_file)

    if len(test_classes_str) > 1:
        logger.info("Evaluating using zero-shot classes only")

        if args.predict_into_folder is not None:
            predict_into_file = os.path.join(args.predict_into_folder, "predictions_zero_shot.txt")

        zero_shot_only_metrics = cat.evaluation_utils.evaluate_model_on_subset(
            model=model,
            dataset_str=data["test"],
            test_classes_str=test_classes_str,
            text_tokenizer=test_dataloader.dataset.text_tokenizer,
            label_tokenizer=test_dataloader.dataset.label_tokenizer,
            predict_into_file=predict_into_file,
            device=args.device,
        )

        if args.predict_into_folder is not None:
            wandb.save(predict_into_file)

        wandb.log({f"zero_shot_only_{k}": v for k, v in zero_shot_only_metrics.items()})

        logger.info("Evaluating using multi-shot classes only")
        if args.predict_into_folder is not None:
            predict_into_file = os.path.join(
                args.predict_into_folder, "predictions_multi_shot.txt"
            )

        multi_shot_classes = list(set(all_classes_str).difference(set(test_classes_str)))

        multi_shot_only_metrics = cat.evaluation_utils.evaluate_model_on_subset(
            model=model,
            dataset_str=data["test"],
            test_classes_str=multi_shot_classes,
            text_tokenizer=test_dataloader.dataset.text_tokenizer,
            label_tokenizer=test_dataloader.dataset.label_tokenizer,
            predict_into_file=predict_into_file,
            device=args.device,
        )

        if predict_into_file is not None:
            wandb.save(predict_into_file)

        wandb.log({f"multi_shot_only_{k}": v for k, v in multi_shot_only_metrics.items()})

    logger.info("Script finished successfully")


if __name__ == "__main__":
    args = parse_args()
    main(args)
