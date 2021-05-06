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


def parse_args(args=None):
    parser = argparse.ArgumentParser()

    # fmt: off
    # Data
    parser.add_argument("--dataset", required=True,
                        help="Name or path to a HuggingFace Datasets dataset")

    parser.add_argument("--load-from-checkpoint", type=str, required=True,
                        help="Checkpoint path is a .pt file storing dict with keys `model_args` to build the model "
                             "architecture and `model_state_dict` to load weights")
    parser.add_argument("--text-field", default=None, type=str)
    parser.add_argument("--class-field", default=None, type=str)
    parser.add_argument("--max-text-length", default=512, type=int)
    parser.add_argument("--evaluate-on", default="validation", type=str,
                        help="a split name to evaluate the model on")

    # --- Evaluation
    parser.add_argument("--batch-size", default=32, type=int)

    # --- Misc
    parser.add_argument("--fp16", default=False, action="store_true")
    parser.add_argument("--predict-into-folder", default=None, type=str,
                        help="Specify this to save predictions into a bunch of files in this folder.")
    parser.add_argument("--n-workers", default=8, type=int)
    parser.add_argument("--tags", default=None)

    args = parser.parse_args(args)
    # fmt: on

    args.tags = args.tags.split(",") if args.tags else []

    args.text_field, args.class_field = cat.utils.infer_field_names(
        args.dataset, text_field=args.text_field, class_field=args.class_field
    )

    logger.info(f"Loading architecture arguments form the checkpoint {args.load_from_checkpoint}")
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

    return args


def main(args):
    accelerator = Accelerator(fp16=args.fp16)

    wandb.init(project="class_attention_eval", config=args, tags=args.tags)
    logger.info(f"Starting the script with the arguments \n{json.dumps(vars(args), indent=4)}")

    logger.info("Creating dataloaders")

    (
        _,
        test_dataloader,
        all_classes_str,
        test_classes_str,
        data,
        _,
    ) = cat.training_utils.prepare_dataloaders(
        dataset_name_or_path=args.dataset,
        model_name=args.model,
        batch_size=args.batch_size,
        num_workers=args.n_workers,
        text_field=args.text_field,
        class_field=args.class_field,
        test_set_name=args.evaluate_on,
        max_text_length=args.max_text_length,
        verbose=True,
    )

    wandb.config.update(
        {"test_classes": ", ".join(sorted(test_classes_str))}, allow_val_change=True
    )

    cat.training_utils.validate_dataloader(test_dataloader, test_classes_str, is_test=True)
    logger.info(f"List of classes in the test set: {', '.join(test_classes_str)}")

    if len(test_classes_str) < 2:
        logger.warning(f"Less than two zero-shot classes")

    # Model
    logger.info("Loading pretrained models")

    text_encoder = transformers.AutoModel.from_pretrained(args.model)
    label_encoder = transformers.AutoModel.from_pretrained(args.model)

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

    model, test_dataloader = accelerator.prepare(model, test_dataloader)

    wandb.log({"model_description": wandb.Html(cat.utils.monospace_html(repr(model)))})

    predict_into_file = None
    if args.predict_into_folder is not None:
        os.makedirs(args.predict_into_folder, exist_ok=True)
        predict_into_file = os.path.join(args.predict_into_folder, "predictions_all_classes.txt")

    logger.info("Evaluating using all classes")

    metrics = cat.evaluation_utils.evaluate_model(
        model=model,
        dataloader=test_dataloader,
        labels_str=all_classes_str,
        zeroshot_labels=test_classes_str,
    )

    if wandb.run is not None:
        wandb.log(metrics)

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
        accelerator=accelerator,
    )

    if args.predict_into_folder is not None:
        wandb.save(predict_into_file)

    wandb.log({f"zero_shot_only_{k}": v for k, v in zero_shot_only_metrics.items()})

    logger.info("Evaluating using multi-shot classes only")

    if args.predict_into_folder is not None:
        predict_into_file = os.path.join(args.predict_into_folder, "predictions_multi_shot.txt")

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
        accelerator=accelerator,
    )

    if predict_into_file is not None:
        wandb.save(predict_into_file)

    wandb.log({f"multi_shot_only_{k}": v for k, v in multi_shot_only_metrics.items()})

    logger.info("Script finished successfully")


if __name__ == "__main__":
    args = parse_args()
    main(args)
