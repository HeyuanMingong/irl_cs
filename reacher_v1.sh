#!/bin/bash

### train in the original environment, original task: goal [0.1,0.1]
echo -e "\nTraining in the original environment, given the task [0.1, 0.1]"
python main.py --env ReacherDyna-v1 --num_iter 500 \
    --lr 0.01 --algorithm reinforce --batch_size 50 --stage pretrain \
    --output output/reacher_v1 --model_path saves/reacher_v1 \
    --task 0.1 0.1 --num_workers 16 --device cuda:4


#######################################################################
echo -e "\nThe environment changes to a new one as [0, 0.14], first trial"
python main.py --env ReacherDyna-v1 --stage finetune --task 0 0.14 \
    --num_iter 500  --batch_size 50 --lr 0.01 --algorithm reinforce \
    --output output/reacher_v1 --model_path saves/reacher_v1 \
    --random --pretrained --prpg --priw \
    --num_workers 16 --device cuda:4


