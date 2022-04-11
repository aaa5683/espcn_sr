import os
import sys
import cv2
from tqdm import tqdm
import shutil
import random as rnd
from tensorflow import keras
from tensorflow.keras.preprocessing import image_dataset_from_directory

def split_data_by_use(logger, train_all=False, prop={'train':0.6,'valid':0.2,'test':0.2}):
    logger.info('and then, split frames by use(train, valid, test)')
    frames_base_path = 'data/frames'

    dir_list = os.listdir(frames_base_path)
    dir_exist_list = {i:(i in dir_list) for i in ['train', 'valid', 'test']}
    if any(list(dir_exist_list.values())) :
        logger.error(f'[ERROR] exist dir : {dir_exist_list}')
        sys.exit(1)
    else:
        for i in ['train', 'valid', 'test']:
            os.mkdir(os.path.join(frames_base_path, i))

    frames_list = [f for f in os.listdir(frames_base_path) if f.endswith('.jpg')]
    rnd.seed(777)
    if train_all:
        train_frames_list = frames_list
        valid_frames_list = rnd.choices(population=frames_list, k=int(len(frames_list) * prop['valid']))
        test_frames_list = rnd.choices(population=frames_list, k=int(len(frames_list) * prop['valid']))
    else:
        train_frames_list = rnd.choices(population=frames_list, k=int(len(frames_list) * prop['test']))
        valid_frames_list = rnd.choices(population=list(set(frames_list)-set(train_frames_list)), k=int(len(frames_list) * prop['valid']))
        test_frames_list = list(set(frames_list)-set(train_frames_list)-set(valid_frames_list))

    print(f'split data to train({len(train_frames_list)}) & valid({len(valid_frames_list)} & test({len(test_frames_list)})')
    for frame_name in tqdm(frames_list):
        if frame_name in train_frames_list:
            shutil.move(src=os.path.join(frames_base_path, frame_name), dst=os.path.join(frames_base_path, 'train'))
        elif frame_name in valid_frames_list:
            shutil.move(src=os.path.join(frames_base_path, frame_name), dst=os.path.join(frames_base_path, 'valid'))
        else:
            shutil.move(src=os.path.join(frames_base_path, frame_name), dst=os.path.join(frames_base_path, 'test'))

def extract_frame_from_video(logger, video_path='data/videos'):
    frame_base_path = 'data/frames'
    if not os.path.exists(frame_base_path):
        os.mkdir(frame_base_path)

    video_name_list = [f for f in os.listdir(video_path) if f.endswith('.mp4')]
    logger.info(f'Total Video Count : {len(video_name_list)}')
    for video_name in video_name_list:
        logger.info(f'{video_name} =============')
        video = cv2.VideoCapture(os.path.join(video_path, video_name))

        if (video.isOpened() == False):
            logger.error("[ERROR] when opening video stream or file")

        total_frame_cnt = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        for f in tqdm(range(total_frame_cnt)):
            ret, frame = video.read()

            if ret == True:
                # cv2.imshow('Frame', frame)
                # Press Q on keyboard to  exit
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    break
                cv2.imwrite(os.path.join(frame_base_path, f'frame_{str(f).zfill(len(str(total_frame_cnt))+1)}.jpg'), frame)
            else:
                break
        video.release()

        # Closes all the frames
        # cv2.destroyAllWindows()

    split_data_by_use(logger)

def get_data_from_url(exists_data=True):
    data_dir = 'BSR'
    root_dir = os.path.join(data_dir, "BSDS500/data")

    if not exists_data:
        dataset_url = "http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/BSR/BSR_bsds500.tgz"
        data_dir = keras.utils.get_file(origin=dataset_url, fname="BSR", untar=True)

    return root_dir

def create_dataset(logger, crop_size=300, upscale_factor=3, batch_size=8):
    frames_base_path = 'data/frames'
    input_size = crop_size // upscale_factor

    train_ds = image_dataset_from_directory(
        os.path.join(frames_base_path, 'train'),
        batch_size=batch_size,
        image_size=(crop_size, crop_size),
        validation_split=0.2,
        subset="training",
        seed=777,
        label_mode=None,
    )

    valid_ds = image_dataset_from_directory(
        os.path.join(frames_base_path, 'valid'),
        batch_size=batch_size,
        image_size=(crop_size, crop_size),
        validation_split=0.2,
        subset="validation",
        seed=777,
        label_mode=None,
    )

    return input_size, train_ds, valid_ds


