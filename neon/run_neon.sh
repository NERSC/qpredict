#!/bin/bash

source ../load_modules.sh

export BATCH_SIZE=100
export EVAL_FREQ=10


for NUM_EPOCHS in 6500
do
    if [ "$1" == "--resume" ]; then 
	    python train_mlp.py  --checkpoint_restart jobwait_model_checkpoint.prm --reset_epochs -o jobwait_output.hd5 --activation_function='ReLU' --initial_learning_rate 1.e-4 -e $NUM_EPOCHS --batch_size $BATCH_SIZE --eval_freq $EVAL_FREQ
    else
	    python train_mlp.py -o jobwait_output.hd5 --activation_function='ReLU' --initial_learning_rate 1.e-4 -e $NUM_EPOCHS --batch_size $BATCH_SIZE --eval_freq $EVAL_FREQ
    fi
done 
