import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
import numpy as np
import pdb

#model_path = '/data/julianlu/Experiment/multimodal_emotion/exp/3cls_bot_bank/pb_model'
model_path = '/data/julianlu/Experiment/multimodal_emotion/exp/3cls_mix_bal_WordEmb_train_L32/pb_model'
model_path = 'models/mymodel_20220407_1040'

max_len = 32

frozen_out_path = 'tf1_model'
frozen_graph_filename = "frozen_graph"
os.makedirs(frozen_out_path, exist_ok = True)

model = tf.keras.models.load_model(model_path)


#Convert Keras model to ConcreteFunction
full_model = tf.function(lambda x: model(x))

# @tf.function()
# def full_model(inputs):
#     pred = model(inputs)
#     return {"predict_probs": pred}

# my_signatures = full_model.get_concrete_function(
#     tf.TensorSpec((None, max_len), tf.int32, name = 'seq_wordids'))# Get frozen ConcreteFunction

# tf.saved_model.save(model, export_dir = frozen_out_path, signatures = my_signatures)


full_model = full_model.get_concrete_function(
    tf.TensorSpec((None, max_len), tf.int32, name = 'seq_wordids'))# Get frozen ConcreteFunction
frozen_func = convert_variables_to_constants_v2(full_model)
frozen_func.graph.as_graph_def()
layers = [op.name for op in frozen_func.graph.get_operations()]
print("-" * 60)
print("Frozen model layers: ")
for layer in layers:
    print(layer)

print("-" * 60)
print("Frozen model inputs: ")
print(frozen_func.inputs)
print("Frozen model outputs: ")
print(frozen_func.outputs)

# Save frozen graph to disk
tf.io.write_graph(graph_or_graph_def=frozen_func.graph,
                  logdir=frozen_out_path,
                  name=f"{frozen_graph_filename}.pb",
                  as_text=False)# Save its text representation
tf.io.write_graph(graph_or_graph_def=frozen_func.graph,
                  logdir=frozen_out_path,
                  name=f"{frozen_graph_filename}.pbtxt",
                  as_text=True)

import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, array_to_img, img_to_array
from modeling import get_model, ESPCNCallback, get_lowres_image, plot_results, upscale_image
import os
md = tf.keras.models.load_model('models/mymodel_20220411_1302')
frames_path = os.listdir('frame')
upscale_factor = 3
total_bicubic_psnr = 0.0
total_test_psnr = 0.0
for index, frame_path in enumerate(frames_path):
    frame_path = os.path.join('frame', frame_path)
    img = load_img(frame_path)
    lowres_input = get_lowres_image(img, upscale_factor)
    w = lowres_input.size[0] * upscale_factor
    h = lowres_input.size[1] * upscale_factor
    highres_img = img.resize((w, h))
    prediction = upscale_image(md, lowres_input)
    lowres_img = lowres_input.resize((w, h))
    lowres_img_arr = img_to_array(lowres_img)
    highres_img_arr = img_to_array(highres_img)
    predict_img_arr = img_to_array(prediction)
    bicubic_psnr = tf.image.psnr(lowres_img_arr, highres_img_arr, max_val=255)
    test_psnr = tf.image.psnr(predict_img_arr, highres_img_arr, max_val=255)

    total_bicubic_psnr += bicubic_psnr
    total_test_psnr += test_psnr

    print(f'\n{frame_path}')
    print(f'PSNR of low resolution image and high resolution image is {bicubic_psnr}')
    print(f'PSNR of predict and high resolution is {test_psnr}')
    plot_results(lowres_img, frame_path, "low")
    plot_results(highres_img, frame_path, "high")
    plot_results(prediction, frame_path, "prediction")
