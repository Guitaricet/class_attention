# experiment set 8 on multiple datasets
# trying out cos2 regularization
# report:
export TOKENIZERS_PARALLELISM=false
cd ..

MODEL=distilbert-base-uncased
SAVE_TO=models/tmp.pth
TAG=es8
EVAL_EVERY=500


for DATA in data/emotion_v0 data/emotion_v1 data/news-category-random-split data/news-category-semantic-split data/topic_v0 data/topic_v1 data/emotion_v0_toneless data/emotion_v1_toneless
do

  python main.py \
    --dataset $DATA \
    --model $MODEL \
    --batch-size 32 \
    --lr 1e-5 \
    --max-epochs 10 \
    --early-stopping 5 \
    --eval-every-steps $EVAL_EVERY \
    --save-to $SAVE_TO \
    --tags $TAG \
    --class-cos2-reg 1.0 \


  rm $SAVE_TO

done
