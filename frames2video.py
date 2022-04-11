import os
import cv2
import numpy as np

def main(logger):
    frames_path = 'data/frames/train'
    for root, dirs, files in os.walk(frames_path, topdown=False):
        bIsFirst = True
        for file_name in files:
            cur_file = os.path.join(frames_path, file_name)
            cur_img = cv2.imread(cur_file)

            logger.info(f'Currently {cur_file} being processed...')
            if (type(cur_img) == np.ndarray):
                if (bIsFirst):
                    frame_height = cur_img.shape[0]
                    frame_width = cur_img.shape[1]

                    # Define the codec and create VideoWriter object.The output is stored in 'outpy.avi' file.
                    video_file = os.path.join(PATH_TO_OUTPUT_VIDEO_DIR, VIDEO_FILE)
                    out = cv2.VideoWriter(video_file, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10, (frame_width, frame_height))

                    # record the current image frame to video file
                    out.write(cur_img)

                    bIsFirst = False

    # When everything done, release the video capture and video write objects
    out.release()