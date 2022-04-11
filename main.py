import warnings
warnings.simplefilter('ignore')
import argparse
import tensorflow as tf
import os
from datetime import datetime
from tensorflow import keras
from tensorflow.keras.preprocessing.image import load_img, array_to_img, img_to_array
from IPython.display import display
from logger import CreateLogger
from prepare_dataset import extract_frame_from_video, create_dataset
from preprocess import scaling, process_input, process_target
from modeling import get_model, ESPCNCallback, get_lowres_image, plot_results, upscale_image


def main(args):
    logger = CreateLogger(logger_name='train', loggfile_path='./log/train.log')
    logger.info('First of all, you need to put videos to train in directory data/videos.')

    if args.extract_frame:
        logger.info('0. Extract frames form videos to train and split them by use.')
        extract_frame_from_video(logger=logger)


    logger.info('1. Create Dataset from frames.')
    upscale_factor = args.upscale_factor
    input_size, train_ds, valid_ds = create_dataset(logger, upscale_factor=upscale_factor)


    # dataset = os.path.join(root_dir, "images")
    test_path = 'data/frames/test'
    test_img_paths = sorted([os.path.join(test_path, f) for f in os.listdir(test_path) if f.endswith('.jpg')])


    logger.info('2. Preprocess Dataset')
    logger.info('2-1. Scaling Dataset : from (0, 255) to (0, 1)')
    train_ds = train_ds.map(scaling)
    valid_ds = valid_ds.map(scaling)

    # for batch in train_ds.take(1):
    #     for img in batch:
    #         display(array_to_img(img))

    train_ds = train_ds.map(lambda x: (process_input(x, input_size), process_target(x)))
    train_ds = train_ds.prefetch(buffer_size=32)

    valid_ds = valid_ds.map(lambda x: (process_input(x, input_size,), process_target(x)))
    valid_ds = valid_ds.prefetch(buffer_size=32)

    # for batch in train_ds.take(1):
    #     for img in batch[0]:
    #         display(array_to_img(img))
    #     for img in batch[1]:
    #         display(array_to_img(img))

    logger.info('3. Model Training')
    early_stopping_callback = keras.callbacks.EarlyStopping(monitor="loss", patience=5)

    checkpoint_filepath = 'log/checkpoint'
    model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=True,
        monitor="loss",
        mode="min",
        save_best_only=True,
    )

    model = get_model(upscale_factor=upscale_factor, channels=1)
    model.summary()

    callbacks = [ESPCNCallback(test_img_paths, upscale_factor), early_stopping_callback, model_checkpoint_callback]
    loss_fn = keras.losses.MeanSquaredError()
    optimizer = keras.optimizers.Adam(learning_rate=0.001)

    epochs = args.epochs
    model.compile(
        optimizer=optimizer,
        loss=loss_fn,
    )
    model.fit(train_ds, epochs=epochs, callbacks=callbacks, validation_data=valid_ds)#, verbose=2)

    logger.info('4. Test')
    # The model weights (that are considered the best) are loaded into the model.
    model.load_weights(checkpoint_filepath)

    total_bicubic_psnr = 0.0
    total_test_psnr = 0.0

    for index, test_img_path in enumerate(test_img_paths):
        img = load_img(test_img_path)
        lowres_input = get_lowres_image(img, upscale_factor)
        w = lowres_input.size[0] * upscale_factor
        h = lowres_input.size[1] * upscale_factor
        highres_img = img.resize((w, h))
        prediction = upscale_image(model, lowres_input)
        lowres_img = lowres_input.resize((w, h))
        lowres_img_arr = img_to_array(lowres_img)
        highres_img_arr = img_to_array(highres_img)
        predict_img_arr = img_to_array(prediction)
        bicubic_psnr = tf.image.psnr(lowres_img_arr, highres_img_arr, max_val=255)
        test_psnr = tf.image.psnr(predict_img_arr, highres_img_arr, max_val=255)

        total_bicubic_psnr += bicubic_psnr
        total_test_psnr += test_psnr

        logger.info(f'PSNR of low resolution image and high resolution image is {bicubic_psnr}')
        logger.info(f'PSNR of predict and high resolution is {test_psnr}')
        plot_results(lowres_img, test_img_path.split('/')[-1], "low")
        plot_results(highres_img, test_img_path.split('/')[-1], "high")
        plot_results(prediction, test_img_path.split('/')[-1], "prediction")

    logger.info("Avg. PSNR of lowres images is %.4f" % (total_bicubic_psnr / 10))
    logger.info("Avg. PSNR of reconstructions is %.4f" % (total_test_psnr / 10))

    model.save(f'models/{args.model_nm}_{datetime.now ().strftime("%Y%m%d_%H%M")}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--extract_frame', default=1, type=int)
    parser.add_argument('--upscale_factor', default=3, type=int)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--model_nm', default='mymodel', type=str)
    args = parser.parse_args()

    main(args)