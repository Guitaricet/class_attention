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
    parser.add_argument("--text-field", default=None, type=str)
    parser.add_argument("--class-field", default=None, type=str)

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
    parser.add_argument("--n-projection-layers", default=None, type=int,
                        help="transform text embedding and class embedding using FCN with this many layers; "
                             "nonlinearity is not used if n=1")
    parser.add_argument("--cross-attention-layers", default=None, type=int,
                        help="apply cross-attention transformer blocks after projection layers "
                             "to make texts and labels interact")
    parser.add_argument("--cross-attention-heads", default=1, type=int,
                        help="number of heads to use in cross-attention")

    parser.add_argument("--no-bias", default=False, action="store_true",
                        help="do not use bias in added layers")
    parser.add_argument("--glove", default=None, type=str,
                        help="path to GloVe embeddings. Use them instead of transformer for class encoding.")
    parser.add_argument("--random-cls-vectors", default=False, action="store_true",
                        help="use random vectors as class vectors (aka untrained word2vec), "
                             "vector size equals to --hidden-size, used as a sanity check / baseline.")

    # training
    parser.add_argument("--max-epochs", default=10, type=int)
    parser.add_argument("--batch-size", default=32, type=int)
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--dropout", default=0.1, type=float)
    parser.add_argument("--freeze-projections", default=False, action="store_true",
                        help="do not train cls_out and txt_out")
    parser.add_argument("--freeze-cls-network", default=False, action="store_true")
    parser.add_argument("--freeze-txt-network", default=False, action="store_true")
    parser.add_argument("--freeze-cls-embeddings", default=False, action="store_true")
    parser.add_argument("--share-txt-cls-network-params", default=False, action="store_true")
    parser.add_argument("--p-extra-classes", default=0, type=float,
                        help="proportion of extra classes to feed into the model during of training at every batch")
    parser.add_argument("--p-no-class", default=0, type=float,
                        help="proportion of labels to randomly drop in the batch")
    parser.add_argument("--early-stopping", default=None, type=int)
    parser.add_argument("--examples-entropy-reg", default=None, type=float,
                        help="maximize the entropy of the predicted distribution on unknown **examples**")
    parser.add_argument("--classes-entropy-reg", default=None, type=float,
                        help="maximize the entropy of the predicted distribution on unknown **classes**")
    parser.add_argument("--regularize-with-real-classes", default=False, action="store_true",
                        help="use real zero-shot classes to maximize the entropy of P(Zero|x_Multi). "
                             "Not practical, serves as oracle/sanity check.")
    parser.add_argument("--classes-entropy-batch-size", default=None, type=int,
                        help="the number of extra classes to compute the regularization term")
    parser.add_argument("--eval-every-steps", default=None, type=int,
                        help="evaluate model each --eval-every-steps steps; does not affect early stopping")
    parser.add_argument("--label-smoothing", default=None, type=float)
    parser.add_argument("--evaluate-on", default="validation", type=str,
                        help="a split name to evaluate the model on")

    # misc
    parser.add_argument("--device", default=None)
    parser.add_argument("--debug", default=False, action="store_true",
                        help="overrides the arguments for a faster run (smaller model, smaller dataset)")
    parser.add_argument("--predict-into-folder", default=None, type=str,
                        help="Specify this to save predictions into a bunch of files in this folder.")
    parser.add_argument("--tags", default=None)
    parser.add_argument("--n-workers", default=8, type=int)
    parser.add_argument("--save-to", default=None, type=str,
                        help="Checkpoint file to save the model state, optimizer state and script arguments")
    parser.add_argument("--wandb-name", default=None, type=str,
                        help="wandb run name")

    args = parser.parse_args(args)
    args.device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    args.tags = args.tags.split(",") if args.tags else []

    # fmt: on

    args.use_bias = not args.no_bias

    if args.debug:
        if args.save_to is not None:
            raise ValueError("--save-to is not supported in debug mode")

        logger.info(
            "Running in a debug mode, overriding dataset, tags, max_epochs, dataset_frac, and test_class_frac args"
        )
        args.dataset = DATASET
        args.dataset_frac = 0.001
        args.test_class_frac = 0.2
        args.max_epochs = 2
        args.tags = ["debug"]

    if args.random_cls_vectors and args.glove is not None:
        raise ValueError("do not provide both --glove and --random-cls-vector")

    if args.random_cls_vectors is not None and args.hidden_size is None:
        raise ValueError("you should provide --hidden-size with --random-cls-vectors")

    if args.classes_entropy_reg is not None:
        if args.glove is None:
            raise NotImplementedError("--classes-entropy-reg is only supported in --glove mode")

    return args


def main(args):
    text_field, class_field = cat.utils.infer_field_names(
        args.dataset, text_field=args.text_field, class_field=args.class_field
    )
    args.text_field = text_field
    args.class_field = class_field

    wandb.init(project="class_attention", config=args, tags=args.tags, name=args.wandb_name)
    logger.info(f"Starting the script with the arguments \n{json.dumps(vars(args), indent=4)}")

    logger.info("Creating dataloaders")
    # TODO: use validation dataset as an unlabeled data source
    (
        train_dataloader,
        test_dataloader,
        all_classes_str,
        test_classes_str,
        data,
        zero_shot_dataloader,  # only examples, no labels (but all examples belong to the test classes set)
    ) = cat.training_utils.prepare_dataloaders(
        dataset_name_or_path=args.dataset,
        model_name=args.model,
        test_class_frac=args.test_class_frac,
        dataset_frac=args.dataset_frac,
        batch_size=args.batch_size,
        p_no_class=args.p_no_class,
        p_extra_classes=args.p_extra_classes,
        num_workers=args.n_workers,
        return_zero_shot_examples=True,
        glove_path=args.glove,
        text_field=text_field,
        class_field=class_field,
        test_set_name=args.evaluate_on,
    )

    wandb.config.update(
        {"test_classes": ", ".join(sorted(test_classes_str))}, allow_val_change=True
    )

    cat.training_utils.validate_dataloader(test_dataloader, test_classes_str, is_test=True)
    cat.training_utils.validate_dataloader(train_dataloader, test_classes_str, is_test=False)

    logger.info(f"List of classes in the test set: {', '.join(test_classes_str)}")

    if len(test_classes_str) < 2:
        logger.warning(f"Less than two zero-shot classes")

    # Model
    logger.info("Creating model and optimizer")

    if args.debug:
        text_encoder = cat.modelling_utils.get_small_transformer()
        label_encoder = cat.modelling_utils.get_small_transformer()
    else:
        text_encoder = transformers.AutoModel.from_pretrained(args.model)
        label_encoder = cat.training_utils.make_label_encoder(
            model_name_or_path=args.model, glove=args.glove
        )

    if args.random_cls_vectors:
        label_encoder = cat.modelling_utils.get_small_transformer(hidden_size=args.hidden_size)

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

    predict_into_file = None
    if args.predict_into_folder is not None:
        os.makedirs(args.predict_into_folder, exist_ok=True)
        predict_into_file = os.path.join(args.predict_into_folder, "predictions_all_classes.txt")

    extra_kwargs = dict()
    if args.examples_entropy_reg:
        extra_kwargs["extra_examples_dataloader"] = zero_shot_dataloader
        extra_kwargs["examples_entropy_reg"] = args.examples_entropy_reg
    if args.classes_entropy_reg:
        assert isinstance(label_encoder, cat.modelling.PreTrainedEmbeddingEncoder)
        # fmt: off
        extra_kwargs["extra_classes_dataloader"] = cat.training_utils.make_extra_classes_dataloader_from_glove(
            glove_path=args.glove,
            batch_size=args.classes_entropy_batch_size or args.batch_size // 2,
            class_names=all_classes_str if args.regularize_with_real_classes else None,
        )
        # fmt: on
        extra_kwargs["classes_entropy_reg"] = args.classes_entropy_reg

    logger.info("Starting training")

    cat.training_utils.train_cat_model(
        model=model,
        model_optimizer=optimizer,
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        all_classes_str=all_classes_str,
        test_classes_str=test_classes_str,
        max_epochs=args.max_epochs,
        device=args.device,
        predict_into_file=predict_into_file,
        early_stopping=args.early_stopping,
        save_path=args.save_to,
        eval_every_steps=args.eval_every_steps,
        label_smoothing=args.label_smoothing,
        **extra_kwargs,
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
            text_field=args.text_field,
            class_field=args.class_field,
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
            text_field=args.text_field,
            class_field=args.class_field,
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
