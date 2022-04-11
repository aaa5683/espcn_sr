#!/bin/bash

MODEL_NM=$1

. venv/bin/activate

python3 main.py --model_nm=${MODEL_NM}

# nohup bash main.sh iu1_360p_134 &