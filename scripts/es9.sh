# experiment set 9
# figuring out the parameters for the adversarial regularization
# report: https://wandb.ai/guitaricet/class_attention/sweeps/4y6j9jls?workspace=user-guitaricet
export TOKENIZERS_PARALLELISM=false
cd ..

#MODEL=distilbert-base-uncased
#SAVE_TO=models/tmp.pth
#TAG=es9
#EVAL_EVERY=500
#UPDATE_FREQ=4
#
#
#for ADV_REG in 0.1 0.05 0.01 1.0
#do
#
#python main.py \
#  --dataset data/emotion_v0 \
#  --model $MODEL \
#  --batch-size 32 \
#  --lr 1e-5 \
#  --max-epochs 10 \
#  --early-stopping 5 \
#  --adv-reg-weight $ADV_REG \
#  --discriminator-update-freq $UPDATE_FREQ \
#  --extra-classes-file data/tfidf_nouns_16k.txt \
#  --eval-every-steps $EVAL_EVERY \
#  --save-to $SAVE_TO \
#  --tags $TAG \
#
#
#rm $SAVE_TO
#
#done


method: bayes
metric:
  goal: maximize
  name: zero_shot_eval/F1_weighted
parameters:
  batch-size:
    value: 32
  dataset:
    value: data/emotion_v0
  early-stopping:
    value: 5
  eval-every-steps:
    value: 500
  extra-classes-file:
    value: data/tfidf_nouns_16k.txt
  adv-reg-weight:
    distribution: categorical
    values:
    - 2
    - 1
    - 0.5
    - 0.2
    - 0.1
    - 0.05
    - 0.01
    - 0.005
    - 0.001
    - 0.0001
  discriminator-update-freq:
    distribution: categorical
    values:
    - 2
    - 3
    - 4
    - 5
    - 6
    - 7
  lr:
    distribution: categorical
    values:
    - 1e-6
    - 3e-6
    - 5e-6
    - 1e-5
    - 3e-5
    - 5e-5
    - 1e-4
    - 2e-4
  max-epochs:
    value: 15
  model:
    value: distilbert-base-uncased
  save-to:
    value: models/es9-tmp.pt
  tags:
    value: es9-sweep
program: main.py
