# experiment set 12
# Trying different layers
# report: https://wandb.ai/guitaricet/class_attention/reports/Per-layer--Vmlldzo2MTc4NDI
export TOKENIZERS_PARALLELISM=false
cd ..

MODEL=distilbert-base-uncased
SAVE_TO=models/tmp.pth
TAG=es12
EVAL_EVERY=500


for LAYER in 0 1 2 3 4 5 6
do

python main.py \
  --dataset data/emotion_v0 \
  --model $MODEL \
  --batch-size 32 \
  --lr 1e-5 \
  --max-epochs 10 \
  --early-stopping 5 \
  --representation-layer $LAYER \
  --eval-every-steps $EVAL_EVERY \
  --save-to $SAVE_TO \
  --tags $TAG \


rm $SAVE_TO

done


for LAYER in 0 1 2 3 4 5 6
do

python main.py \
  --dataset data/emotion_v1 \
  --model $MODEL \
  --batch-size 32 \
  --lr 1e-5 \
  --max-epochs 10 \
  --early-stopping 5 \
  --representation-layer $LAYER \
  --eval-every-steps $EVAL_EVERY \
  --save-to $SAVE_TO \
  --tags $TAG \


rm $SAVE_TO

done


for LAYER in 0 1 2 3 4 5 6
do

python main.py \
  --dataset data/news-category-random-split \
  --model $MODEL \
  --batch-size 32 \
  --lr 1e-5 \
  --max-epochs 10 \
  --early-stopping 5 \
  --representation-layer $LAYER \
  --eval-every-steps $EVAL_EVERY \
  --save-to $SAVE_TO \
  --tags $TAG \


rm $SAVE_TO

done


for LAYER in 0 1 2 3 4 5 6
do

python main.py \
  --dataset data/news-category-semantic-split \
  --model $MODEL \
  --batch-size 32 \
  --lr 1e-5 \
  --max-epochs 10 \
  --early-stopping 5 \
  --representation-layer $LAYER \
  --eval-every-steps $EVAL_EVERY \
  --save-to $SAVE_TO \
  --tags $TAG \


rm $SAVE_TO

done
