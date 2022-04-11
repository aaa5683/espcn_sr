#!/bin/bash

start_ts=`date`

echo "Start at ${start_ts} ================="

youtube-dl sqgxcCjD04s -f 137 -o ./iu1_1080p_137.mp4

ffmpeg -i iu1_1080p_137.mp4 -to 216 new.mp4

rm iu1_1080p_137.mp4

mv new.mp4 iu1_1080p_137.mp4

echo "iu1_1080p_137.mp4 done."


youtube-dl sqgxcCjD04s -f 399 -o ./iu1_1080p_399.mp4

ffmpeg -i iu1_1080p_399.mp4 -to 216 new.mp4

rm iu1_1080p_399.mp4

mv new.mp4 iu1_1080p_399.mp4

echo "iu1_1080p_399.mp4 done."


youtube-dl sqgxcCjD04s -f 134 -o ./iu1_360p_134.mp4

ffmpeg -i iu1_360p_134.mp4 -to 216 new.mp4

rm iu1_360p_134.mp4

mv new.mp4 iu1_360p_134.mp4

echo "iu1_360p_134.mp4 done."


youtube-dl sqgxcCjD04s -f 396 -o ./iu1_360p_396.mp4

ffmpeg -i iu1_360p_396.mp4 -to 216 new.mp4

rm iu1_360p_396.mp4

mv new.mp4 iu1_360p_396.mp4

echo "iu1_360p_396.mp4 done."


youtube-dl 3iM_06QeZi8 -f 137 -o ./iu2_1080p_137.mp4

ffmpeg -i iu2_1080p_137.mp4 -ss 3 -to 210 new.mp4

rm iu2_1080p_137.mp4

mv new.mp4 iu2_1080p_137.mp4

echo "iu2_1080p_137.mp4 done."


youtube-dl 3iM_06QeZi8 -f 399 -o ./iu2_1080p_399.mp4

ffmpeg -i iu2_1080p_399.mp4 -ss 3 -to 210 new.mp4

rm iu2_1080p_399.mp4

mv new.mp4 iu2_1080p_399.mp4

echo "iu2_1080p_399.mp4 done."


youtube-dl 3iM_06QeZi8 -f 134 -o ./iu2_360p_134.mp4

ffmpeg -i iu2_360p_134.mp4 -ss 3 -to 210 new.mp4

rm iu2_360p_134.mp4

mv new.mp4 iu2_360p_134.mp4

echo "iu2_360p_134.mp4 done."


youtube-dl 3iM_06QeZi8 -f 396 -o ./iu2_360p_396.mp4

ffmpeg -i iu2_360p_396.mp4 -ss 3 -to 210 new.mp4

rm iu2_360p_396.mp4

mv new.mp4 iu2_360p_396.mp4

echo "iu2_360p_396.mp4 done."


end_ts=`date`

echo "All Done at ${end_ts} ================="

# nohup bash get_iu.sh > get_iu.log 2>&1 &
