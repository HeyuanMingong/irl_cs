#!/bin/bash

### train in the original environment, original task: goal [0.1,0.1]
echo "Training in the original environment, given the task [0.1, 0.1]"
python main.py --env ReacherDyna-v1 --num_iter 500 \
    --lr 0.01 --algorithm reinforce --batch_size 50 --stage pretrain \
    --output output/reacher_v1 --model_path saves/reacher_v1 \
    --task 0.1 0.1 --num_workers 4 --device cpu

:<<BLOCK
#######################################################################
echo "The environment changes to a new one as [0, 0.14], first trial"
python main.py --env ReacherDyna-v1 --stage finetune --trial 1 --task 0 0.14 \
    --num_iter 500 --relax_epis 0 --is_eps 1e-2 --batch_size 50 --model_num 3000 \
    --upsilon 1e-2 --nu 0.8 --rmax 200 --psi 1.0 --algorithm reinforce --lr 0.01 \
    --output demo_output/reacher_v1 --model_path demo_saves/reacher_v1 \
    --ran --fine --no-isam --no-wei --iwis --prpg \
    --num_workers 4 --device cpu
BLOCK
