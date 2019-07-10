#!/bin/bash

### change the argument 'task' to set the task in the original environment
### change the device to 'cuda:0' if a GPU is available
echo -e "\nTraining in the original environment, given the task [0.25, 0.25]"
python main.py --env Navigation2D-v1 --num_iter 100 --lr 0.01 --batch_size 20 \
    --algorithm reinforce --stage pretrain --task 0.25 0.25 \
    --output output/navi_v1 --model_path saves/navi_v1 \
    --num_workers 4 --device cpu

### change the argument 'task' to set the task in the new environment
echo -e "\nThe environment changes to a new one as [0, -0.25]..."
python main.py --env Navigation2D-v1 --task 0 -0.25 --num_iter 100 \
    --batch_size 20 --lr 0.01 --algorithm reinforce --stage finetune \
    --random --pretrained --prpg --priw --pr --iw \
    --output output/navi_v1 --model_path saves/navi_v1 \
    --num_workers 4 --device cpu 

