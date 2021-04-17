# experiment set 5 on multiple datasets
# report: https://wandb.ai/guitaricet/class_attention/reports/ES5-Mar-29-Project-Update--Vmlldzo1Njg4MzY
export TOKENIZERS_PARALLELISM=false
cd ..

MODEL=distilbert-base-uncased
SAVE_TO=models/tmp.pth
TAG=es5
EVAL_EVERY=500


python main.py \
  --dataset data/emotion_v0 \
  --model $MODEL \
  --batch-size 32 \
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
  --batch-size 32 \
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
  --batch-size 32 \
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
  --batch-size 32 \
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
  --batch-size 32 \
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
  --batch-size 32 \
  --lr 1e-5 \
  --max-epochs 10 \
  --early-stopping 3 \
  --eval-every-steps $EVAL_EVERY \
  --save-to $SAVE_TO \
  --tags $TAG \


rm $SAVE_TO

# Start these

python main.py \
  --dataset data/emotion_v0_toneless \
  --model $MODEL \
  --batch-size 32 \
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
  --batch-size 32 \
  --lr 1e-5 \
  --max-epochs 10 \
  --early-stopping 3 \
  --eval-every-steps $EVAL_EVERY \
  --save-to $SAVE_TO \
  --tags $TAG \


rm $SAVE_TO
