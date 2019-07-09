#!/bin/bash

### change the argument 'task' to set the task in the original environment
echo "Training in the original environment"
python main.py --env Navigation2D-v2 --num_iter 500 --lr 0.01 --batch_size 20 \
    --algorithm reinforce --stage pretrain \
    --output output/navi_v2 --model_path saves/navi_v2 \
    --task -0.3 0.2 0.25 0.25 0.1 -0.1 --num_workers 4 --device cpu


### change the argument 'task' to set the task in the new environment
echo "The environment changes to a new one..."
python main.py --env Navigation2D-v2  --num_iter 500 \
    --batch_size 20 --lr 0.01 --algorithm reinforce --stage finetune \
    --random --pretrained --prpg --pr --iw --priw \
    --output output/navi_v2 --model_path saves/navi_v2 \
    --task 0.2 -0.2 -0.35 -0.25 0 0.2 --num_workers 4 --device cpu 

