# experiment set 2
# report: https://wandb.ai/guitaricet/class_attention/reports/Snapshot-Mar-12-2021-11-42am--Vmlldzo1MjUyODU
export TOKENIZERS_PARALLELISM=false
cd ..

# big dataset
DATASET=data/news-category-random-split
MODEL=distilbert-base-uncased
SAVE_TO=models/es2.pth
TAG=es2


python main.py \
  --dataset $DATASET \
  --model $MODEL \
  --batch-size 32 \
  --lr 1e-5 \
  --max-epochs 10 \
  --early-stopping 3 \
  --save-to $SAVE_TO \
  --normalize-cls \
  --normalize-txt \
  --tags $TAG \


rm $SAVE_TO

python main.py \
  --dataset $DATASET \
  --model $MODEL \
  --batch-size 32 \
  --lr 1e-5 \
  --max-epochs 10 \
  --early-stopping 3 \
  --save-to $SAVE_TO \
  --normalize-cls \
  --normalize-txt \
  --freeze-cls-network \
  --tags $TAG


rm $SAVE_TO

python main.py \
  --dataset $DATASET \
  --model $MODEL \
  --batch-size 32 \
  --lr 1e-5 \
  --max-epochs 10 \
  --early-stopping 3 \
  --save-to $SAVE_TO \
  --normalize-cls \
  --normalize-txt \
  --cross-attention-layers 1 \
  --tags $TAG


rm $SAVE_TO

python main.py \
  --dataset $DATASET \
  --model $MODEL \
  --batch-size 32 \
  --lr 1e-5 \
  --max-epochs 10 \
  --early-stopping 3 \
  --save-to $SAVE_TO \
  --normalize-cls \
  --normalize-txt \
  --glove data/glove.6B.300d.txt \
  --freeze-cls-network \
  --tags $TAG


rm $SAVE_TO


python main.py \
  --dataset $DATASET \
  --model roberta-base \
  --batch-size 32 \
  --lr 1e-5 \
  --max-epochs 10 \
  --early-stopping 3 \
  --save-to $SAVE_TO \
  --normalize-cls \
  --normalize-txt \
  --tags $TAG,roberta


rm $SAVE_TO


python main.py \
  --dataset $DATASET \
  --model $MODEL \
  --glove data/glove.6B.300d.txt \
  --batch-size 32 \
  --lr 1e-5 \
  --max-epochs 10 \
  --early-stopping 3 \
  --save-to $SAVE_TO \
  --normalize-cls \
  --normalize-txt \
  --tags $TAG,entropy-reg \
  --n-projection-layers 1 \
  --classes-entropy-reg 1.0 \


rm $SAVE_TO


python main.py \
  --dataset $DATASET \
  --model $MODEL \
  --glove data/glove.6B.300d.txt \
  --batch-size 32 \
  --lr 1e-5 \
  --max-epochs 10 \
  --early-stopping 3 \
  --save-to $SAVE_TO \
  --normalize-cls \
  --normalize-txt \
  --tags $TAG,entropy-reg \
  --n-projection-layers 1 \
  --classes-entropy-reg 1.0 \
  --regularize-with-real-classes \


rm $SAVE_TO

python main.py \
  --dataset $DATASET \
  --model $MODEL \
  --glove data/glove.6B.300d.txt \
  --batch-size 32 \
  --lr 1e-5 \
  --max-epochs 10 \
  --early-stopping 3 \
  --save-to $SAVE_TO \
  --normalize-cls \
  --normalize-txt \
  --tags $TAG,entropy-reg \
  --n-projection-layers 1 \
  --examples-entropy-reg 1.0 \


rm $SAVE_TO

python main.py \
  --dataset $DATASET \
  --model $MODEL \
  --glove data/glove.6B.300d.txt \
  --batch-size 32 \
  --lr 1e-5 \
  --max-epochs 10 \
  --early-stopping 3 \
  --save-to $SAVE_TO \
  --normalize-cls \
  --normalize-txt \
  --tags $TAG,entropy-reg \
  --n-projection-layers 1 \
  --classes-entropy-reg 1.0 \
  --examples-entropy-reg 1.0 \


rm $SAVE_TO


python main.py \
  --dataset $DATASET \
  --model $MODEL \
  --batch-size 32 \
  --lr 1e-4 \
  --max-epochs 10 \
  --early-stopping 3 \
  --save-to $SAVE_TO \
  --normalize-cls \
  --normalize-txt \
  --tags $TAG,lr \


rm $SAVE_TO

python main.py \
  --dataset $DATASET \
  --model $MODEL \
  --batch-size 32 \
  --lr 1e-3 \
  --max-epochs 10 \
  --early-stopping 3 \
  --save-to $SAVE_TO \
  --normalize-cls \
  --normalize-txt \
  --tags $TAG,lr \


rm $SAVE_TO


python main.py \
  --dataset $DATASET \
  --model $MODEL \
  --batch-size 32 \
  --lr 1e-6 \
  --max-epochs 10 \
  --early-stopping 3 \
  --save-to $SAVE_TO \
  --normalize-cls \
  --normalize-txt \
  --tags $TAG,lr \


rm $SAVE_TO


python main.py \
  --dataset $DATASET \
  --model $MODEL \
  --batch-size 32 \
  --lr 5e-5 \
  --max-epochs 10 \
  --early-stopping 3 \
  --save-to $SAVE_TO \
  --normalize-cls \
  --normalize-txt \
  --tags $TAG,lr \


rm $SAVE_TO

python main.py \
  --dataset $DATASET \
  --model roberta-base \
  --batch-size 32 \
  --lr 5e-5 \
  --max-epochs 10 \
  --early-stopping 3 \
  --save-to $SAVE_TO \
  --normalize-cls \
  --normalize-txt \
  --tags $TAG,roberta,lr


rm $SAVE_TO
