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
from accelerate import Accelerator

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
    # Data
    parser.add_argument("--dataset", default=DATASET,
                        help="Name or path to a HuggingFace Datasets dataset")
    parser.add_argument("--test-dataset", default=None,
                        help="A dataset to evaluate on (--evaluate_on field), if not specified --dataset is used. "
                             "Use this option if you pretrain on a ranking task (wiki) and evaluate on a "
                             "classification task (news-category/0shot-tc)")
    parser.add_argument("--test-class-frac", default=0.0, type=float,
                        help="a fraction of classes to remove from the training set (and use for zero-shot)")
    parser.add_argument("--dataset-frac", default=1.0, type=float,
                        help="a fraction of dataset to train and evaluate on, used for debugging")
    parser.add_argument("--text-field", default=None, type=str)
    parser.add_argument("--class-field", default=None, type=str)
    parser.add_argument("--test-text-field", default=None, type=str)
    parser.add_argument("--test-class-field", default=None, type=str)
    parser.add_argument("--max-text-length", default=512, type=int)
    parser.add_argument("--evaluate-on", default="validation", type=str,
                        help="a split name to evaluate the model on")
    parser.add_argument("--faiss-index-path", default=None, type=str,
                        help="path to a faiss index for the --dataset, use scripts/index_wikipedia.py to produce it")
    parser.add_argument("--index-field", default="text_emb", type=str,
                        help="dataset field name that is associated with the faiss index")

    # Fine-tuning
    parser.add_argument("--load-from-checkpoint", default=None, type=str,
                        help="If provided, instead of creating a model, it is restored from a checkpoint file. "
                             "Checkpoint path is a .pt file storing dict with keys `model_args` to build the model "
                             "architecture and `model_state_dict` to load weights")

    # Architecture
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
    parser.add_argument("--representation-layer", default=-1, type=int,
                        help="hidden layer representations to use for the text and class representations."
                             "The last layer is used by default (-1). 0 means the first layer.")

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
    parser.add_argument("--random-cls-vectors", default=False, action="store_true",
                        help="use random vectors as class vectors (aka untrained word2vec), "
                             "vector size equals to --hidden-size, used as a sanity check / baseline.")

    # --- Training
    parser.add_argument("--max-epochs", default=10, type=int)
    parser.add_argument("--early-stopping", default=None, type=int)
    parser.add_argument("--early-stopping-metric", default="eval/F1_macro", type=str)
    parser.add_argument("--batch-size", default=32, type=int)
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--dropout", default=0.1, type=float)
    parser.add_argument("--eval-every-steps", default=None, type=int,
                        help="evaluate model each --eval-every-steps steps; does not affect early stopping")
    parser.add_argument("--label-smoothing", default=None, type=float)

    # Layer freezing and parameter sharing
    parser.add_argument("--freeze-projections", default=False, action="store_true",
                        help="do not train cls_out and txt_out")
    parser.add_argument("--freeze-cls-network", default=False, action="store_true")
    parser.add_argument("--freeze-txt-network", default=False, action="store_true")
    parser.add_argument("--freeze-cls-embeddings", default=False, action="store_true")
    parser.add_argument("--share-txt-cls-network-params", default=False, action="store_true")

    # --- Regularization
    # Extra classes and entropy reg
    parser.add_argument("--extra-classes-file", default=None, type=str,
                        help="path to a text file with extra class names (one name on every line), "
                             "used for adversarial regularization")
    parser.add_argument("--extra-classes-batch-size", default=None, type=int,
                        help="the number of extra classes to compute the regularization term")

    # Adversarial
    parser.add_argument("--discriminator-update-freq", default=None, type=int)
    parser.add_argument("--discr-lr", default=None, type=int)
    parser.add_argument("--adv-reg-weight", default=1.0, type=float,
                        help="A regularization strength for the model (not the discriminator) part of the loss")
    parser.add_argument("--wasserstein", default=False, action="store_true",
                        help="Use Wasserstein-style loss for adversarial regularization")
    parser.add_argument("--wasserstein-for-sweeps", default=None, type=int,
                        help="Same as --wasserstein, but can be used with wandb sweeps. 1 for True and 0 for False")
    parser.add_argument("--discriminator-hidden", default=1024, type=int)
    parser.add_argument("--discriminator-layers", default=3, type=int)

    # Other kinds of regularization
    parser.add_argument("--class-cos2-reg", default=None, type=float,
                        help="Add cos^2 between class embeddings to the loss, weight by this value")

    # --- Misc
    parser.add_argument("--fp16", default=False, action="store_true")
    parser.add_argument("--debug", default=False, action="store_true",
                        help="overrides the arguments for a faster run (smaller model, smaller dataset)")
    parser.add_argument("--predict-into-folder", default=None, type=str,
                        help="Specify this to save predictions into a bunch of files in this folder.")
    parser.add_argument("--tags", default=None)
    parser.add_argument("--n-workers", default=8, type=int)
    parser.add_argument("--no-pin-memory", default=False, action="store_true")
    parser.add_argument("--save-to", default=None, type=str,
                        help="Checkpoint file to save the model state, optimizer state and script arguments")
    parser.add_argument("--wandb-name", default=None, type=str,
                        help="wandb run name")

    args = parser.parse_args(args)
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

    if args.discr_lr is None and args.discriminator_update_freq is not None:
        args.discr_lr = args.lr

    if args.wasserstein_for_sweeps is not None:
        args.wasserstein = bool(int(args.wasserstein_for_sweeps))
        args.wasserstein_for_sweeps = None

    args.text_field, args.class_field = cat.utils.infer_field_names(
        args.dataset, text_field=args.text_field, class_field=args.class_field
    )

    if args.test_dataset is not None:
        logger.info(
            f"Evaluation will be performed on the {args.evaluate_on} field of {args.test_dataset} dataset."
        )
        args.test_text_field = args.test_text_field or args.text_field
        args.test_class_field = args.test_class_field or args.class_field

    if args.load_from_checkpoint:
        logger.info(
            "Architecture arguments provided to the script are ignored in case of --finetune-cat-model"
        )
        logger.info("Loading architecture arguments form the checkpoint")
        checkpoint_args = torch.load(args.load_from_checkpoint, map_location="cpu")["model_args"]

        args.model = checkpoint_args["model"]
        args.hidden_size = checkpoint_args["hidden_size"]
        args.normalize_txt = checkpoint_args["normalize_txt"]
        args.normalize_cls = checkpoint_args["normalize_cls"]
        args.scale_attention = checkpoint_args["scale_attention"]
        args.temperature = checkpoint_args["temperature"]
        args.learn_temperature = checkpoint_args["learn_temperature"]
        args.representation_layer = checkpoint_args["representation_layer"]
        args.n_projection_layers = checkpoint_args["n_projection_layers"]
        args.cross_attention_layers = checkpoint_args["cross_attention_layers"]
        args.cross_attention_heads = checkpoint_args["cross_attention_heads"]
        args.no_bias = checkpoint_args["no_bias"]
        args.random_cls_vectors = checkpoint_args["random_cls_vectors"]

    if args.faiss_index_path is not None:
        if not os.path.exists(args.faiss_index_path):
            raise ValueError(f"--faiss-index-path does not exist. Path provided: {args.faiss_index_path}")

    return args


def main(args):
    accelerator = Accelerator(fp16=args.fp16)

    wandb.init(project="class_attention", config=args, tags=args.tags, name=args.wandb_name)
    logger.info(f"Starting the script with the arguments \n{json.dumps(vars(args), indent=4)}")

    logger.info("Creating dataloaders")

    (
        train_dataloader,
        test_dataloader,
        all_classes_str,
        test_classes_str,
        data,
        ranking_test_dataloader,
    ) = cat.training_utils.prepare_dataloaders(
        dataset_name_or_path=args.dataset,
        model_name=args.model,
        test_class_frac=args.test_class_frac,
        dataset_frac=args.dataset_frac,
        batch_size=args.batch_size,
        num_workers=args.n_workers,
        text_field=args.text_field,
        class_field=args.class_field,
        test_set_name=args.evaluate_on,
        test_dataset_name_or_path=args.test_dataset,
        test_text_field=args.test_text_field,
        test_class_field=args.test_class_field,
        max_text_length=args.max_text_length,
        verbose=True,
        faiss_index_path=args.faiss_index_path,
        index_field=args.index_field,
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
    logger.info("Loading pretrained models")

    if args.debug:
        text_encoder = cat.modelling_utils.get_small_transformer()
        label_encoder = cat.modelling_utils.get_small_transformer()
    else:
        text_encoder = transformers.AutoModel.from_pretrained(args.model)
        label_encoder = transformers.AutoModel.from_pretrained(args.model)

    if args.random_cls_vectors:
        label_encoder = cat.modelling_utils.get_small_transformer(hidden_size=args.hidden_size)

    model = cat.ClassAttentionModel(
        text_encoder,
        label_encoder,
        **vars(args),
    )

    if args.load_from_checkpoint:
        logger.info(
            f"Loading class attention weights from a checkpoint {args.load_from_checkpoint}"
        )
        model.load_state_dict_from_checkpoint(args.load_from_checkpoint)

    parameters = model.get_trainable_parameters()
    optimizer = torch.optim.Adam(parameters, lr=args.lr)

    model, optimizer, train_dataloader, test_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, test_dataloader
    )

    if ranking_test_dataloader is not None:
        ranking_test_dataloader = accelerator.prepare_data_loader(ranking_test_dataloader)

    discriminator = None
    discriminator_optimizer = None
    if args.discriminator_update_freq is not None:
        discriminator = cat.modelling_utils.make_mlp(
            n_layers=args.discriminator_layers,
            input_size=model.final_hidden_size,
            hidden_size=args.discriminator_hidden,
            output_size=1,
            spectral_normalization=True,
        )
        discriminator_optimizer = torch.optim.Adam(
            discriminator.parameters(), lr=args.discr_lr, betas=(0.0, 0.9)
        )

        discriminator, discriminator_optimizer = accelerator.prepare(
            discriminator, discriminator_optimizer
        )

    wandb.watch(model, log="all")
    wandb.log({"model_description": wandb.Html(cat.utils.monospace_html(repr(model)))})

    predict_into_file = None
    if args.predict_into_folder is not None:
        os.makedirs(args.predict_into_folder, exist_ok=True)
        predict_into_file = os.path.join(args.predict_into_folder, "predictions_all_classes.txt")

    extra_kwargs = dict()
    if args.extra_classes_file:
        # fmt: off
        extra_kwargs["extra_classes_dataloader"] = cat.training_utils.make_extra_classes_dataloader_from_file(
            file_path=args.extra_classes_file,
            tokenizer=test_dataloader.dataset.label_tokenizer,
            batch_size=args.extra_classes_batch_size or args.batch_size // 2,
        )
        # fmt: on

    logger.info("Starting training")

    cat.training_utils.train_cat_model(
        model=model,
        model_optimizer=optimizer,
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        all_classes_str=all_classes_str,
        test_classes_str=test_classes_str,
        max_epochs=args.max_epochs,
        accelerator=accelerator,
        predict_into_file=predict_into_file,
        early_stopping=args.early_stopping,
        early_stopping_metric=args.early_stopping_metric,
        save_path=args.save_to,
        eval_every_steps=args.eval_every_steps,
        label_smoothing=args.label_smoothing,
        discriminator=discriminator,
        discriminator_optimizer=discriminator_optimizer,
        discriminator_update_freq=args.discriminator_update_freq,
        class_cos2_reg=args.class_cos2_reg,
        adv_reg_weight=args.adv_reg_weight,
        use_wasserstein_loss=args.wasserstein,
        ranking_test_dataloader=ranking_test_dataloader,
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
            text_field=args.test_text_field or args.text_field,
            class_field=args.test_class_field or args.class_field,
            test_classes_str=test_classes_str,
            text_tokenizer=test_dataloader.dataset.text_tokenizer,
            label_tokenizer=test_dataloader.dataset.label_tokenizer,
            predict_into_file=predict_into_file,
            accelerator=accelerator,
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
            text_field=args.test_text_field or args.text_field,
            class_field=args.test_class_field or args.class_field,
            test_classes_str=multi_shot_classes,
            text_tokenizer=test_dataloader.dataset.text_tokenizer,
            label_tokenizer=test_dataloader.dataset.label_tokenizer,
            predict_into_file=predict_into_file,
            accelerator=accelerator,
        )

        if predict_into_file is not None:
            wandb.save(predict_into_file)

        wandb.log({f"multi_shot_only_{k}": v for k, v in multi_shot_only_metrics.items()})

    logger.info("Script finished successfully")


if __name__ == "__main__":
    args = parse_args()
    main(args)
