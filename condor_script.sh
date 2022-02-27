#!/bin/bash

source ~/.bashrc
conda activate count_2


python3 ./syn_exps.py > syn_exps.out
python3 ./sem_exps.py > sem_exps.out