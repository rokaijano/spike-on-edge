"""Implementations of layers/models used in EfficientDet."""

import tensorflow as tf
import numpy as np
from model.utils import filter_detections

import tensorflow_addons as tfa


def ClassDetector(inputs,
             num_classes=80,
             embedding_size = 32,
             channels=64,
             num_anchors=9,
             depth=3,
             kernel_size=3,
             depth_multiplier=1,
             include_cls_top=True,
             name='class_det'):
    ###Initialize classification model.


    main_layer = tf.keras.layers.SeparableConv2D

    bias_init = tf.constant_initializer(-np.log((1 - 0.01) / 0.01))

    embeddings = main_layer(
        embedding_size * num_anchors,
        kernel_size,
        padding='same',
        depth_multiplier=depth_multiplier,
        activation=None,
        pointwise_initializer=tf.initializers.variance_scaling(),
        depthwise_initializer=tf.initializers.variance_scaling(),
        bias_initializer=bias_init,
        name=f'{name}_embeddings'
    )

    detection = main_layer(
        num_classes * num_anchors,
        kernel_size,
        padding='same',
        depth_multiplier=depth_multiplier,
        activation=None,
        pointwise_initializer=tf.initializers.variance_scaling(),
        depthwise_initializer=tf.initializers.variance_scaling(),
        bias_initializer=bias_init,
        name=f'{name}_detection'
    )

    
    x = inputs

    for i in range(depth):
        x = main_layer(
            channels,
            kernel_size,
            padding='same',
            depth_multiplier=depth_multiplier,
            pointwise_initializer=tf.initializers.variance_scaling(),
            depthwise_initializer=tf.initializers.variance_scaling(),
            bias_initializer=tf.zeros_initializer(),
            name=f'{name}_separable_conv_{i}')(x)
        x = tf.keras.layers.BatchNormalization(name=f'{name}_bn_{i}')(x)
        x = tf.keras.layers.Activation(tf.nn.silu)(x)
    

    if include_cls_top:

        det = detection(x)

        det = tf.keras.layers.Reshape([-1, num_classes])(det)
        det = tf.keras.layers.Softmax()(det)


        #################################
        #################################
        embed = embeddings(x)
        embed = tf.keras.layers.Reshape([-1, embedding_size])(embed)
        embed = tf.keras.layers.Dense(embedding_size, kernel_initializer=tf.keras.initializers.HeNormal())(embed)
        embed = tf.keras.layers.BatchNormalization()(embed)
        embed = tf.keras.layers.ReLU()(embed)
        embed = tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=-1))(embed)
        embed = tf.keras.layers.Reshape([-1, embedding_size])(embed)
        
        return embed, det

    return x


def BoxRegressor(inputs,
             channels=64,
             num_anchors=9,
             depth=3,
             kernel_size=3,
             depth_multiplier=1,
             name='box_regressor'):
    #Initialize regression model.

    main_layer = tf.keras.layers.SeparableConv2D
    
    boxes = main_layer(
            4 * num_anchors,
            kernel_size,
            padding='same',
            depth_multiplier=depth_multiplier,
            activation=None,
            pointwise_initializer=tf.initializers.variance_scaling(),
            depthwise_initializer=tf.initializers.variance_scaling(),
            bias_initializer=tf.zeros_initializer(),
            name=f'{name}_box_preds'
        )
    
    for i in range(depth):
        inputs = main_layer(
            channels,
            kernel_size,
            padding='same',
            depth_multiplier=depth_multiplier,
            pointwise_initializer=tf.initializers.variance_scaling(),
            depthwise_initializer=tf.initializers.variance_scaling(),
            bias_initializer=tf.zeros_initializer(),
            name=f'{name}_separable_conv_{i}')(inputs)
        inputs = tf.keras.layers.BatchNormalization(name=f'{name}_bn_{i}')(inputs)
        inputs = tf.keras.layers.Activation(tf.nn.silu)(inputs)
    box_output = boxes(inputs)

    return box_output
