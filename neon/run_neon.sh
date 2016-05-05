#!/bin/bash

source ../load_modules.sh

export BATCH_SIZE=100
export EVAL_FREQ=10

for NUM_EPOCHS in 30000
do
    python train_mlp.py -o jobwait_output.hd5 --activation_function='ReLU' --initial_learning_rate 1.e-4 -e $NUM_EPOCHS --batch_size $BATCH_SIZE --eval_freq $EVAL_FREQ
done 
