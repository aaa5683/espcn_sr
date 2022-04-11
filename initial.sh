#!/bin/bash

VIDEO=$1

rm -rf log predict_test_images data

mkdir -p log predict_test_images data/videos

cp iu_videos/${VIDEO} data/videos

python3 -m venv venv

. venv/bin/activate

pip install -U pip

pip install -r requirements.txt

# bash initial.sh iu1_360p_134.mp4