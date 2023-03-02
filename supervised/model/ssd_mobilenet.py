import tensorflow as tf
import tensorflow_model_optimization as tfmot
import tensorflow_addons as tfa

from .mobilenet_v2 import MobileNetV2
from .layers import ClassDetector, BoxRegressor

def SSDMobileNetV2(
             input_shape,
             channels=64,
             num_classes=80,
             embedding_size=32,
             num_anchors=18,
             heads_depth=3,
             class_kernel_size=3,
             class_depth_multiplier=1,
             box_kernel_size=3,
             box_depth_multiplier=1,
             quant_aware = False,
             include_cls_top=True):


    backbone_weight_file = "backbone_weights/"

    inputs = tf.keras.layers.Input(shape=input_shape)
    
    reshape_layer_boxes = tf.keras.layers.Reshape([-1, 4])

    bns_first1 = tf.keras.layers.BatchNormalization(beta_initializer="random_uniform", gamma_initializer="random_uniform")
    bns_first2 = tf.keras.layers.BatchNormalization(beta_initializer="random_uniform", gamma_initializer="random_uniform", axis=[0,1])

    bns = [
        tf.keras.layers.BatchNormalization(beta_initializer="random_uniform", gamma_initializer="random_uniform") for i in range(2)
    ]
    concat = tf.keras.layers.Concatenate(axis=-1)

    concat_1 = tf.keras.layers.Concatenate(axis=1)
    concat_2 = tf.keras.layers.Concatenate(axis=1)

    x = inputs

    features = MobileNetV2(alpha=0.2, include_top=False, weights=None, input_shape=input_shape, input_tensor = x)
    
    embed, detection = ClassDetector(features,
                                   channels=channels,
                                   num_classes=num_classes,
                                   embedding_size = embedding_size,
                                   num_anchors=num_anchors,
                                   depth=heads_depth,
                                   kernel_size=class_kernel_size,
                                   depth_multiplier=class_depth_multiplier,
                                   use_positional_encoding=False,
                                   include_cls_top=include_cls_top)

    boxes = BoxRegressor(features,
                                channels=channels,
                                num_anchors=num_anchors,
                                depth=heads_depth,
                                kernel_size=box_kernel_size,
                                depth_multiplier=box_depth_multiplier)


    boxes = reshape_layer_boxes(boxes)
    
    final_model = tf.keras.Model(inputs, [boxes, embed, detection])

    if quant_aware:
        final_model = tfmot.quantization.keras.quantize_model(final_model)

    return final_model


