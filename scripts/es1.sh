export TOKENIZERS_PARALLELISM=false

DATASET=data/news-category-random-split-small

python scripts/run_baselines --dataset $DATASET
python main.py --dataset data/news-category-random-split-small --normalize-txt --normalize-cls --learn-temperature --tags ES1
python main.py --dataset data/news-category-random-split-small --normalize-txt --normalize-cls --learn-temperature --tags ES1 --max-epochs 30
python main.py --dataset data/news-category-random-split-small --normalize-txt --normalize-cls --temperature -2 --tags ES1 --max-epochs 30
python main.py --dataset data/news-category-random-split-small --scale-attention --learn-temperature --tags ES1 --max-epochs 30
python main.py \
  --dataset data/news-category-random-split-small \
  --normalize-txt --normalize-cls --learn-temperature \
  --attention-type bahdanau --bahdanau-layers 2 --hidden 128 \
  --tags ES1 --max-epochs 30 \

