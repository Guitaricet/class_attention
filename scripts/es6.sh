# experiment set 6 on multiple datasets
# same as ES5, but with BERT-base instead of DistilBERT
# report:
set -e
export TOKENIZERS_PARALLELISM=false
cd ..

MODEL=bert-base-uncased
SAVE_TO=models/tmp.pth
TAG=es6
EVAL_EVERY=500
BATCH_SIZE=24


python main.py \
  --dataset data/emotion_v0 \
  --model $MODEL \
  --batch-size $BATCH_SIZE \
  --lr 1e-5 \
  --max-epochs 10 \
  --early-stopping 3 \
  --eval-every-steps $EVAL_EVERY \
  --save-to $SAVE_TO \
  --tags $TAG \


rm $SAVE_TO


python main.py \
  --dataset data/emotion_v1 \
  --model $MODEL \
  --batch-size $BATCH_SIZE \
  --lr 1e-5 \
  --max-epochs 10 \
  --early-stopping 3 \
  --eval-every-steps $EVAL_EVERY \
  --save-to $SAVE_TO \
  --tags $TAG \


rm $SAVE_TO


python main.py \
  --dataset data/emotion_v0_toneless \
  --model $MODEL \
  --batch-size $BATCH_SIZE \
  --lr 1e-5 \
  --max-epochs 10 \
  --early-stopping 3 \
  --eval-every-steps $EVAL_EVERY \
  --save-to $SAVE_TO \
  --tags $TAG \


rm $SAVE_TO


python main.py \
  --dataset data/emotion_v1_toneless \
  --model $MODEL \
  --batch-size $BATCH_SIZE \
  --lr 1e-5 \
  --max-epochs 10 \
  --early-stopping 3 \
  --eval-every-steps $EVAL_EVERY \
  --save-to $SAVE_TO \
  --tags $TAG \


rm $SAVE_TO


python main.py \
  --dataset data/news-category-random-split \
  --model $MODEL \
  --batch-size $BATCH_SIZE \
  --lr 1e-5 \
  --max-epochs 10 \
  --early-stopping 3 \
  --eval-every-steps $EVAL_EVERY \
  --save-to $SAVE_TO \
  --tags $TAG \


rm $SAVE_TO


python main.py \
  --dataset data/news-category-semantic-split \
  --model $MODEL \
  --batch-size $BATCH_SIZE \
  --lr 1e-5 \
  --max-epochs 10 \
  --early-stopping 3 \
  --eval-every-steps $EVAL_EVERY \
  --save-to $SAVE_TO \
  --tags $TAG \


rm $SAVE_TO


python main.py \
  --dataset data/news-category-semantic-split \
  --model $MODEL \
  --batch-size $BATCH_SIZE \
  --lr 1e-5 \
  --max-epochs 10 \
  --early-stopping 3 \
  --eval-every-steps $EVAL_EVERY \
  --save-to $SAVE_TO \
  --tags $TAG \


rm $SAVE_TO


python main.py \
  --dataset data/topic_v0 \
  --model $MODEL \
  --batch-size $BATCH_SIZE \
  --lr 1e-5 \
  --max-epochs 10 \
  --early-stopping 3 \
  --eval-every-steps $EVAL_EVERY \
  --save-to $SAVE_TO \
  --tags $TAG \


rm $SAVE_TO
