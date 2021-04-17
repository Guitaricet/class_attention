# experiment set 4 on the emotion dataset (V0 and V1)
# sweep: https://wandb.ai/guitaricet/class_attention/sweeps/v7g4njtb
export TOKENIZERS_PARALLELISM=false
cd ..

echo "Emotion V0"

# big dataset
NAME=emotion_v0

DATASET=data/$NAME
MODEL=distilbert-base-uncased
SAVE_TO=models/es3.pth
TAG=es3-$NAME
EVAL_EVERY=3000


for P_EXTRA in 0 0.01 0.1 0.2 0.5
do
  for P_NO in 0 0.01 0.1
  do

  python main.py \
    --dataset $DATASET \
    --model $MODEL \
    --batch-size 32 \
    --lr 1e-5 \
    --max-epochs 10 \
    --early-stopping 3 \
    --eval-every-steps $EVAL_EVERY \
    --p-extra-classes $P_EXTRA \
    --p-no-class $P_NO \
    --save-to $SAVE_TO \
    --tags $TAG \


  rm $SAVE_TO

  done
done


echo "Emotion V0 is finished"
echo "Starting Emotion V1"

# big dataset
NAME=emotion_v1

DATASET=data/$NAME
MODEL=distilbert-base-uncased
SAVE_TO=models/es3.pth
TAG=es3-$NAME
EVAL_EVERY=3000


for P_EXTRA in 0 0.01 0.1 0.2 0.5
do
  for P_NO in 0 0.01 0.1
  do

  python main.py \
    --dataset $DATASET \
    --model $MODEL \
    --batch-size 32 \
    --lr 1e-5 \
    --max-epochs 10 \
    --early-stopping 3 \
    --eval-every-steps $EVAL_EVERY \
    --p-extra-classes $P_EXTRA \
    --p-no-class $P_NO \
    --save-to $SAVE_TO \
    --tags $TAG \


  rm $SAVE_TO

  done
done


# SWEEP
#program: main.py
#method: bayes
#metric:
#  goal: maximize
#  name: zero_shot_eval/F1_weighted
#early_terminate:
#  type: hyperband
#  min_iter: 3
#parameters:
#  dataset:
#    value: data/emotion_v0
#  model:
#    value: distilbert-base-uncased
#  batch-size:
#    value: 32
#  lr:
#    value: 1e-5
#  max-epochs:
#    value: 10
#  early-stopping:
#    value: 3
#  eval-every-steps:
#    value: 2500
#  save-to:
#    value: models/es4_emotion_v0.pt
#  tags:
#    value: es4-emo0-sweep
#  p-extra-classes:
#    distribution: categorical
#    values:
#      - 0.0
#      - 0.001
#      - 0.01
#      - 0.05
#      - 0.1
#      - 0.15
#      - 0.2
#      - 0.25
#      - 0.3
#      - 0.5
#      - 0.6
#      - 0.7
#      - 0.8
#      - 0.9
#      - 1.0
#  p-no-class:
#    distribution: categorical
#    values:
#      - 0.0
#      - 0.001
#      - 0.01
#      - 0.05
#      - 0.1
#      - 0.15
#      - 0.2
#      - 0.25
#      - 0.3
#      - 0.5
