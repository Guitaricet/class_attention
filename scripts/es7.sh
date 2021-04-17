# experiment set 7 on multiple datasets
# trying out adversarial regularization
# report:
export TOKENIZERS_PARALLELISM=false
cd ..

MODEL=distilbert-base-uncased
SAVE_TO=models/tmp.pth
TAG=es7-grad-clip
EVAL_EVERY=500
UPDATE_FREQ=4


python main.py \
  --dataset data/emotion_v0 \
  --model $MODEL \
  --batch-size 32 \
  --lr 1e-5 \
  --max-epochs 10 \
  --early-stopping 5 \
  --discriminator-update-freq $UPDATE_FREQ \
  --extra-classes-file data/tfidf_nouns_16k.txt \
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
  --early-stopping 5 \
  --discriminator-update-freq $UPDATE_FREQ \
  --extra-classes-file data/tfidf_nouns_16k.txt \
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
  --early-stopping 5 \
  --discriminator-update-freq $UPDATE_FREQ \
  --extra-classes-file data/tfidf_nouns_16k.txt \
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
  --early-stopping 5 \
  --discriminator-update-freq $UPDATE_FREQ \
  --extra-classes-file data/tfidf_nouns_16k.txt \
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
  --early-stopping 5 \
  --discriminator-update-freq $UPDATE_FREQ \
  --extra-classes-file data/tfidf_nouns_16k.txt \
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
  --early-stopping 5 \
  --discriminator-update-freq $UPDATE_FREQ \
  --extra-classes-file data/tfidf_nouns_16k.txt \
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
  --early-stopping 5 \
  --discriminator-update-freq $UPDATE_FREQ \
  --extra-classes-file data/tfidf_nouns_16k.txt \
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
  --early-stopping 5 \
  --discriminator-update-freq $UPDATE_FREQ \
  --extra-classes-file data/tfidf_nouns_16k.txt \
  --eval-every-steps $EVAL_EVERY \
  --save-to $SAVE_TO \
  --tags $TAG \


rm $SAVE_TO
