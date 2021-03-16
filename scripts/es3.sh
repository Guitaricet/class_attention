# experiment set 3
# report: https://wandb.ai/guitaricet/class_attention/reports/ES3-Mar-16-Project-Update--Vmlldzo1MzYzOTY
export TOKENIZERS_PARALLELISM=false
cd ..

# big dataset
DATASET=data/news-category-random-split
MODEL=distilbert-base-uncased
SAVE_TO=models/es3.pth
TAG=es3
EVAL_EVERY=3000


python main.py \
  --dataset $DATASET \
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
  --dataset $DATASET \
  --model $MODEL \
  --batch-size 32 \
  --lr 1e-5 \
  --max-epochs 10 \
  --early-stopping 3 \
  --normalize-cls \
  --eval-every-steps $EVAL_EVERY \
  --save-to $SAVE_TO \
  --tags $TAG,normalization \


rm $SAVE_TO


python main.py \
  --dataset $DATASET \
  --model $MODEL \
  --batch-size 32 \
  --lr 1e-5 \
  --max-epochs 10 \
  --early-stopping 3 \
  --normalize-txt \
  --eval-every-steps $EVAL_EVERY \
  --save-to $SAVE_TO \
  --tags $TAG,normalization \


rm $SAVE_TO


python main.py \
  --dataset $DATASET \
  --model $MODEL \
  --batch-size 32 \
  --lr 1e-5 \
  --max-epochs 10 \
  --early-stopping 3 \
  --normalize-cls \
  --normalize-txt \
  --eval-every-steps $EVAL_EVERY \
  --save-to $SAVE_TO \
  --tags $TAG,normalization \


rm $SAVE_TO



python main.py \
  --dataset $DATASET \
  --model $MODEL \
  --batch-size 32 \
  --lr 1e-5 \
  --max-epochs 10 \
  --early-stopping 3 \
  --n-projection-layers 1 \
  --hidden 300 \
  --freeze-projections \
  --glove data/glove.6B.300d.txt \
  --eval-every-steps $EVAL_EVERY \
  --save-to $SAVE_TO \
  --tags $TAG,glove \


rm $SAVE_TO


python main.py \
  --dataset $DATASET \
  --model $MODEL \
  --batch-size 32 \
  --lr 1e-5 \
  --max-epochs 10 \
  --early-stopping 3 \
  --n-projection-layers 1 \
  --hidden 300 \
  --glove data/glove.6B.300d.txt \
  --eval-every-steps $EVAL_EVERY \
  --save-to $SAVE_TO \
  --tags $TAG,glove \


rm $SAVE_TO


python main.py \
  --dataset $DATASET \
  --model $MODEL \
  --batch-size 32 \
  --lr 1e-5 \
  --max-epochs 10 \
  --early-stopping 3 \
  --n-projection-layers 1 \
  --eval-every-steps $EVAL_EVERY \
  --save-to $SAVE_TO \
  --tags $TAG,projections \


rm $SAVE_TO


python main.py \
  --dataset $DATASET \
  --model $MODEL \
  --batch-size 32 \
  --lr 1e-5 \
  --max-epochs 10 \
  --early-stopping 3 \
  --n-projection-layers 1 \
  --freeze-projections \
  --eval-every-steps $EVAL_EVERY \
  --save-to $SAVE_TO \
  --tags $TAG,projections \


rm $SAVE_TO


python main.py \
  --dataset $DATASET \
  --model $MODEL \
  --batch-size 32 \
  --lr 1e-5 \
  --max-epochs 10 \
  --early-stopping 3 \
  --n-projection-layers 2 \
  --eval-every-steps $EVAL_EVERY \
  --save-to $SAVE_TO \
  --tags $TAG,projections \


rm $SAVE_TO


python main.py \
  --dataset $DATASET \
  --model $MODEL \
  --batch-size 32 \
  --lr 1e-5 \
  --max-epochs 10 \
  --early-stopping 3 \
  --cross-attention-layers 1 \
  --n-projection-layers 1 \
  --hidden 128 \
  --eval-every-steps $EVAL_EVERY \
  --save-to $SAVE_TO \
  --tags $TAG,cross-attention \


rm $SAVE_TO


python main.py \
  --dataset $DATASET \
  --model $MODEL \
  --batch-size 32 \
  --lr 1e-5 \
  --max-epochs 10 \
  --early-stopping 3 \
  --cross-attention-layers 1 \
  --n-projection-layers 1 \
  --hidden 512 \
  --eval-every-steps $EVAL_EVERY \
  --save-to $SAVE_TO \
  --tags $TAG,cross-attention \


rm $SAVE_TO


python main.py \
  --dataset $DATASET \
  --model $MODEL \
  --batch-size 32 \
  --lr 1e-5 \
  --max-epochs 10 \
  --early-stopping 3 \
  --learn-temperature \
  --eval-every-steps $EVAL_EVERY \
  --save-to $SAVE_TO \
  --tags $TAG,temperature \


rm $SAVE_TO


python main.py \
  --dataset $DATASET \
  --model $MODEL \
  --batch-size 32 \
  --lr 1e-5 \
  --max-epochs 10 \
  --early-stopping 3 \
  --p-training-classes 1.0 \
  --eval-every-steps $EVAL_EVERY \
  --save-to $SAVE_TO \
  --tags $TAG,all_classes \


rm $SAVE_TO


python main.py \
  --dataset $DATASET \
  --model $MODEL \
  --batch-size 32 \
  --lr 1e-5 \
  --max-epochs 10 \
  --early-stopping 3 \
  --n-projection-layers 1 \
  --hidden 300 \
  --glove data/glove.6B.300d.txt \
  --examples-entropy-reg 1.0 \
  --eval-every-steps $EVAL_EVERY \
  --save-to $SAVE_TO \
  --tags $TAG,entropy-glove \


rm $SAVE_TO


python main.py \
  --dataset $DATASET \
  --model $MODEL \
  --batch-size 32 \
  --lr 1e-5 \
  --max-epochs 10 \
  --early-stopping 3 \
  --n-projection-layers 1 \
  --hidden 300 \
  --glove data/glove.6B.300d.txt \
  --classes-entropy-reg 1.0 \
  --eval-every-steps $EVAL_EVERY \
  --save-to $SAVE_TO \
  --tags $TAG,entropy-glove \


rm $SAVE_TO


python main.py \
  --dataset $DATASET \
  --model $MODEL \
  --batch-size 32 \
  --lr 1e-5 \
  --max-epochs 10 \
  --early-stopping 3 \
  --n-projection-layers 1 \
  --hidden 300 \
  --glove data/glove.6B.300d.txt \
  --examples-entropy-reg 1.0 \
  --classes-entropy-reg 1.0 \
  --eval-every-steps $EVAL_EVERY \
  --save-to $SAVE_TO \
  --tags $TAG,entropy-glove \


rm $SAVE_TO

echo "Script finished"
