# experiment set 10
# Wasserstein GAN and spectral normalization of the discriminator parameters
# report:
export TOKENIZERS_PARALLELISM=false
cd ..

MODEL=distilbert-base-uncased
SAVE_TO=models/tmp.pth
TAG=es10
EVAL_EVERY=500
UPDATE_FREQ=4


for ADV_REG in 0.1 0.05 0.01 1.0
do

python main.py \
  --dataset data/emotion_v0 \
  --model $MODEL \
  --batch-size 32 \
  --lr 1e-5 \
  --max-epochs 10 \
  --early-stopping 5 \
  --adv-reg-weight $ADV_REG \
  --discriminator-update-freq $UPDATE_FREQ \
  --extra-classes-file data/tfidf_nouns_16k.txt \
  --wasserstein \
  --eval-every-steps $EVAL_EVERY \
  --save-to $SAVE_TO \
  --tags $TAG \


rm $SAVE_TO

done
