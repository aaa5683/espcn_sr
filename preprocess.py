import tensorflow as tf

# Scale from (0, 255) to (0, 1)
def scaling(input_image):
    input_image = input_image / 255.0
    return input_image

# Use TF Ops to process.
def process_input(input, input_size):
    input = tf.image.rgb_to_yuv(input)
    last_dimension_axis = len(input.shape) - 1
    y, u, v = tf.split(input, 3, axis=last_dimension_axis)
    return tf.image.resize(y, [input_size, input_size], method="area")

def process_target(input):
    input = tf.image.rgb_to_yuv(input)
    last_dimension_axis = len(input.shape) - 1
    y, u, v = tf.split(input, 3, axis=last_dimension_axis)
    return y
