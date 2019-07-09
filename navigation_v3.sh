#!/bin/bash

### change the argument 'task' to set the task in the original environment
echo "Training in the original environment"
python main.py --env Navigation2D-v3 --stage pretrain \
    --lr 0.01 --algorithm reinforce --batch_size 20 --num_iter 100 \
    --output output/navi_v3 --model_path saves/navi_v3 \
    --task -0.25 0.25 0 -0.25 0.25 0.25 0.5 0.5 --num_workers 4 --device cpu


### change the argument 'task' to set the task in the new environment
echo "The environment changes to a new one..."
python main.py --env Navigation2D-v3 --stage finetune \
    --num_iter 100  --batch_size 20 --algorithm reinforce --lr 0.01 \
    --random --pretrained --prpg --pr --iw --priw \
    --output output/navi_v3 --model_path saves/navi_v3 \
    --task 0 -0.25 0.25 0 -0.25 -0.25 0 -0.5 --num_workers 4 --device cpu 

