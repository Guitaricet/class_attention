# Class Attention

Slides: [google doc link](https://docs.google.com/presentation/d/1u4C4xdlb4ziDpFj_hwA-qdjy2tHs_7p7KCYcqYyzyAc/edit?usp=sharing)

## Installation

```bash
pip install -r requirements.txt
pip install -e .
```

## Data preparation

To generate News-category splits run
```bash
cd scripts
python generate_splits.py
```

## Usage

```bash
main.py

optional arguments:
  --test-class-frac TEST_CLASS_FRAC
                        a fraction of classes to remove from the training set (and use for zero-shot)
  --dataset-frac DATASET_FRAC
                        a fraction of dataset to train and evaluate on, used for debugging
  --model MODEL
  --hidden-size HIDDEN_SIZE
  --normalize-txt
  --normalize-cls
  --scale-attention     we recommend to use scaling if normalization is not used
  --temperature TEMPERATURE
                        softmax temperature (used as the initial value if --learn-temperature
  --learn-temperature   learn the softmax temperature as an additional scalar parameter
  --remove-n-lowest-pc REMOVE_N_LOWEST_PC
                        remove n lowest principal components from the class embeddings
  --use-n-projection-layers USE_N_PROJECTION_LAYERS
                        transform text embedding and class embedding using FCN with this many layers; nonlinearity is not used if n=1
  --attention-type {dot-product,bahdanau}
  --max-epochs MAX_EPOCHS
  --batch-size BATCH_SIZE
  --lr LR
  --freeze-projections  do not train cls_out and txt_out
  --freeze-cls-network
  --share-txt-cls-network-params
  --device DEVICE
  --debug               overrides the arguments for a faster run (smaller model, smaller dataset)
  ```

Usage example:

```bash
python main.py \
  --test-class-frac 0.2 \
  --dataset-frac 0.1 \
  --normalize-cls \

```

### Some scripts we ran

```bash
cd scripts
# run notebooks/25_wikipedia.ipynb to create wikipedia_rank_nochache_May6
python index_wikipedia.py
python tokenize_wikipedia.py --dataset ../data/wikipedia_ranknli_distilroberta_base_v2_encoded --tokenizer distilbert-base-uncased
```