import torch
import datasets
import faiss
from sentence_transformers import SentenceTransformer
torch.set_grad_enabled(False)


print("Loading the model")
model = SentenceTransformer("nli-distilroberta-base-v2")
batch_size = 64
suffix = "nli_distilroberta_base_v2_encoded"

print("Loading the data")
data = datasets.load_from_disk('../data/wikipedia_rank_nocache_May6')

def embed(texts):
    return model.encode(texts, batch_size=batch_size)

def map_fn(x):
    return {"text_emb": embed(x["text"]), "title_emb": embed(x["title"])}

print("Encoding")
data["train"] = data["train"].map(map_fn, batched=True, batch_size=batch_size)

print("Saving data to disk")
data.save_to_disk("../data/wikipedia_rank_" + suffix)

# Index choice:
# transform = "OPQ"
# neighbors = 64  # number of neighbors, M
# reduced_dim = 128  # output dimensionality of the transform
#
# clustering_method = "IVF"  # only IFV is supported on GPU (?)
# n_clusters = 1024  # recommended sqrt(dataset_size) ~= sqrt(10^6) ~= 1024
# clustering_suffix = ""  # maybe you should try IVF65536_HNSW32, but it may not work on GPU (GPU may not fit it anyway)
#
# quantization_type = "PQ"
# n_codes = 64  # number of 4-bit codes, also M (why?)
#
# # uses "fast scan" version of the PQ that relies on SIMD instructions for distance computations.
# # Supports only nbits=4 for now. The suffix _64 indicates the bbs factor used (must be a multiple of 32).
# # The suffix fsr (only for IVF) indicates that the vectors should be encoded by residual (slower, more accurate)
# quantization_suffix = "x4fs"
#
# index_recipe = f"{transform}{neighbors}_{reduced_dim},{clustering_method}{n_clusters}{clustering_suffix},{quantization_type}{n_codes}{quantization_suffix}"
# print("Index recipe: ", index_recipe)
#
# index = faiss.index_factory(512, index_recipe) # "OPQ64_128,IVF1024,PQ64x4f"

print("Building index object")
INDEX_STR = "OPQ64_128,IVF65536_HNSW32,PQ64x4fs"

print("Training index")
data["train"].add_faiss_index(column="text_emb", string_factory=INDEX_STR, train_size=300_000, metric_type=faiss.METRIC_INNER_PRODUCT)

print("Saving index")
data["train"].save_faiss_index("text_emb", f"../data/wikipedia_rank_index_{suffix}.faiss")
