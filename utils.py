import tensorflow as tf
import tensorflow.contrib.slim as slim
import math


def blockLayer(x, channels, r, kernel_size=[3,3]):
    output = tf.layers.conv2d(x, channels, (3, 3), padding='same', dilation_rate=(r, r), use_bias=False)
    return tf.nn.relu(output)

def resDenseBlock(x, channels=64, layers=8, kernel_size=[3,3], scale=1):
    outputs = [x]
    rates = [1,1,1,1,1,1,1,1]
    for i in range(layers):
        output = blockLayer(tf.concat(outputs[:i],3) if i>=1 else x, channels, rates[i])
        outputs.append(output)

    output = tf.concat(outputs, 3)
    output = slim.conv2d(output, channels, [1,1])
    output *= scale
    return x + output

def upsample(x, scale=2, features=64):
    output = x
    if (scale & (scale-1)) == 0:
        for _ in range(int(math.log(scale, 2))):
            output = tf.layers.conv2d(output, 4*features, (3, 3), padding='same', use_bias=False)
            output = pixelshuffle(output, 2)
    elif scale == 3:
        output = tf.layers.conv2d(output, 9*features, (3, 3), padding='same', use_bias=False)
        output = pixelshuffle(output, 3)
    else:
        raise NotImplementedError
    return output

def pixelshuffle(x, upscale_factor):
    return tf.depth_to_space(x, upscale_factor)

"""
Tensorflow log base 10.
Found here: https://github.com/tensorflow/tensorflow/issues/1666
"""
def log10(x):
  numerator = tf.log(x)
  denominator = tf.log(tf.constant(10, dtype=numerator.dtype))
  return numerator / denominator
