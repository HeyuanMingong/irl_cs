#!/bin/bash
:<<BLOCK
### train in the original environment, with physical parameters reacher.xml
echo "Train the original environment in myrllib/envs/mujoco/assets/reacher_0.xml"
python main.py --env ReacherDyna-v2 --num_iter 500 --task 0 \
    --lr 0.01 --algorithm reinforce --stage pretrain --batch_size 50 \
    --output output/reacher_v2 --model_path saves/reacher_v2 \
    --num_workers 16 --device cuda:4
BLOCK

#######################################################################
echo "The environment changes to a new one as reacher_2.xml..."
python main.py --env ReacherDyna-v2 --stage finetune --task 2 \
    --num_iter 500 --batch_size 50 --lr 0.01 --algorithm reinforce \
    --output output/reacher_v2 --model_path saves/reacher_v2 \
    --random --pretrained --prpg --priw \
    --num_workers 16 --device cuda:4

