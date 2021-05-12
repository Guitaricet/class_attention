# Index choice:
# https://github.com/facebookresearch/faiss/wiki/Guidelines-to-choose-an-index
# transform = "OPQ"
# neighbors = 64  # number of neighbors, M
# reduced_dim = 128  # output dimensionality of the transform
#
# clustering_method = "IVF"  # only IFV is supported on GPU (?)
# n_clusters = 1024  # recommended sqrt(dataset_size) ~= 4 * sqrt(5 * 10^6) ~= 8192
# clustering_suffix = ""  # maybe you should try IVF65536_HNSW32, but it may not work on GPU (GPU may not fit it anyway)
#
# quantization_type = "PQ"
# n_codes = 64  # number of 4-bit codes, also M (why?)
#
# uses "fast scan" version of the PQ that relies on SIMD instructions for distance computations.
# Supports only nbits=4 for now. The suffix _64 indicates the bbs factor used (must be a multiple of 32).
# The suffix fsr (only for IVF) indicates that the vectors should be encoded by residual (slower, more accurate)
# quantization_suffix = ""  # because "x4fs" is not supported on GPU
#
# index_recipe = f"{transform}{neighbors}_{reduced_dim},{clustering_method}{n_clusters}{clustering_suffix},{quantization_type}{n_codes}{quantization_suffix}"
# print("Index recipe: ", index_recipe)
#
# index = faiss.index_factory(512, index_recipe) # "OPQ64_128,IVF1024,PQ64"

import argparse
import logging
import os
import sys
from timeit import timeit

import torch
import datasets
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
torch.set_grad_enabled(False)
np.random.seed(42)


logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    stream=sys.stdout,
)
logger = logging.getLogger(os.path.basename(__file__))
logging.getLogger("transformers.configuration_utils").setLevel(logging.WARNING)
logging.getLogger("wandb.sdk.internal.internal").setLevel(logging.WARNING)

MODEL = "nli-distilroberta-base-v2"
INDEX_STR = "OPQ64_128,IVF8192,PQ32"  # PQ64 does not work on 1080TI, because it does not have enough shared memory


def parse_args(args=None):
    parser = argparse.ArgumentParser()

    # fmt: off
    parser.add_argument("--dataset",
                        help="Name or path to a HuggingFace Datasets dataset")
    parser.add_argument("--model", default=MODEL,
                        help="sentence-transformer model to encode texts")
    parser.add_argument("--index-str", default=INDEX_STR,
                        help="fairseq index description for index factory")
    parser.add_argument("--batch-size", default=64, type=int)
    parser.add_argument("--save-to", default=None,
                        help="by default saved to the same directory with a name {dataset}_{model}_encoded "
                             "for the encoded data and {dataset}_{model}_{index_str}.faiss for the index")
    parser.add_argument("--train-size", default=3_000_000, type=int,
                        help="number of dataset points to train the index")
    parser.add_argument("--device", default=None, type=int,
                        help="GPU number to train the index on")
    # fmt: on

    args = parser.parse_args(args)
    return args


if __name__ == '__main__':
    args = parse_args()

    data_save_path = args.save_to or (args.dataset + "_" + args.model + "_encoded")

    if os.path.exists(data_save_path):
        logger.info(f"The data is already encoded at {data_save_path}. "
                    f"Loading this data and skipping the encoding (embedding) step.")
        data = datasets.load_from_disk(data_save_path)

    else:
        logger.info("Loading the model")
        model = SentenceTransformer(args.model)

        logger.info("Loading the data")
        data = datasets.load_from_disk(args.dataset)

        def embed(texts):
            return model.encode(texts, batch_size=args.batch_size, show_progress_bar=False)

        def map_fn(x):
            return {"text_emb": embed(x["text"])}

        logger.info("Encoding")
        data["train"] = data["train"].map(map_fn, batched=True)

        logger.info("Saving data to disk")
        data.save_to_disk(data_save_path)

    logger.info("Training index")
    data["train"].add_faiss_index(
        column="text_emb",
        string_factory=args.index_str,
        train_size=args.train_size,
        metric_type=faiss.METRIC_INNER_PRODUCT,
        device=args.device,
        faiss_verbose=True,
    )

    logger.info("Saving index")

    index_save_path = f"{args.dataset}_{args.model}_{args.index_str}.faiss"
    if args.save_to is not None:
        index_save_path = f"{args.save_to}_{args.index_str}.faiss"

    # workaround for https://github.com/huggingface/datasets/issues/2350
    if args.device is not None:
        data["train"]._indexes["text_emb"].faiss_index = faiss.index_gpu_to_cpu(data["train"]._indexes["text_emb"].faiss_index)

    data["train"].save_faiss_index("text_emb", index_save_path)

    logger.info("Benchmarking the speed")

    def benchmark_fn():
        random_vector = np.random.randn(768).astype(np.float32)
        data["train"].get_nearest_examples("text_emb", random_vector, k=10)

    benchmark_out = timeit("benchmark_fn()", number=100, globals=globals())
    logger.info(benchmark_out)

    logger.info("Script finished successfully")
