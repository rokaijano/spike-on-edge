"""EfficientDet losses

This script contains implementations of Focal loss for classification task and
Huber loss for regression task. Also script includes composition of these losses
for quick setup of training pipeline.
"""

import tensorflow as tf

from tensorflow.keras.losses import mse, binary_crossentropy
from tensorflow.keras import backend as K

import tensorflow_addons as tfa
from model.anchors import *
from model.utils import *


class FocalLoss(tf.keras.losses.Loss):
    """Focal loss implementations."""

    def __init__(self,
                 alpha=0.25,
                 gamma=1.5,
                 label_smoothing=0.1,
                 name='focal_loss'):
        """Initialize parameters for Focal loss.

        FL = - alpha_t * (1 - p_t) ** gamma * log(p_t)
        This implementation also includes label smoothing for preventing overconfidence.
        """
        super().__init__(name=name, reduction="none")
        self.alpha = alpha
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.ce = tf.keras.losses.CategoricalCrossentropy() #

    def call(self, y_true, y_pred):
        """Calculate Focal loss.

        Args:
            y_true: a tensor of ground truth values with
                shape (batch_size, num_anchor_boxes, num_classes).
            y_pred: a tensor of predicted values with
                shape (batch_size, num_anchor_boxes, num_classes).

        Returns:
            A float tensor with shape (batch_size, num_anchor_boxes) with
            loss value for every anchor box.
        """
        #y_pred = tf.cast(y_pred, tf.float32)
        #y_true = tf.cast(y_true, tf.float32)

        prob = y_pred#prob = tf.sigmoid(y_pred)

        pt = y_true * prob + (1 - y_true) * (1 - prob)
        at = y_true * self.alpha + (1 - y_true) * (1 - self.alpha)

        y_true = y_true * (1.0 - self.label_smoothing) + 0.5 * self.label_smoothing

        ce = self.ce(y_true, y_pred)
        #ce = tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true, logits=y_pred) # 

        loss = at * (1.0 - pt)**self.gamma * ce
        return tf.reduce_sum(loss, axis=-1)


class BoxLoss(tf.keras.losses.Loss):
    """Huber loss implementation."""

    def __init__(self,
                 delta=1.0,
                 name='box_loss'):
        super().__init__(name=name, reduction="none")
        self.delta = delta

    def call(self, y_true, y_pred):
        """Calculate Huber loss.

        Args:
            y_true: a tensor of ground truth values with shape (batch_size, num_anchor_boxes, 4).
            y_pred: a tensor of predicted values with shape (batch_size, num_anchor_boxes, 4).

        Returns:
            A float tensor with shape (batch_size, num_anchor_boxes) with
            loss value for every anchor box.
        """

        loss = tf.abs(y_true - y_pred)
        l1 = self.delta * (loss - 0.5 * self.delta)
        l2 = 0.5 * loss ** 2
        box_loss = tf.where(tf.less(loss, self.delta), l2, l1)
        return tf.reduce_sum(box_loss, axis=-1)

class NWD(tf.keras.losses.Loss):

    """
    
    Arxiv: https://arxiv.org/pdf/2110.13389.pdf

    """
    def __init__(self,
                 delta=1.0,
                 name="nwd_loss"):

        super().__init__(name=name, reduction="none")
        self.delta = delta

    def call(self, y_true, y_pred):
        """
        Args:
            y_true: a tensor of ground truth values with shape (batch_size, num_anchor_boxes, 4).
            y_pred: a tensor of predicted values with shape (batch_size, num_anchor_boxes, 4).

        Returns:
            A float tensor with shape (batch_size, num_anchor_boxes) with
            loss value for every anchor box.
        """
        c = 1
        w_dist = tf.math.sqrt((y_true[...,0]-y_pred[..., 0])**2 + (y_true[..., 1]-y_pred[..., 1])**2 + (y_true[..., 2]-y_pred[..., 2])**2 + (y_true[..., 3]-y_pred[..., 3])**2)
        loss = 1 - tf.math.exp(-tf.math.sqrt(w_dist**2)/c)


        l1 = self.delta * (loss - 0.5 * self.delta)
        l2 = 0.5 * loss ** 2
        box_loss = tf.where(tf.less(loss, self.delta), l2, l1)
        return box_loss #tf.reduce_sum(box_loss, axis=-1)

class SSDDetLoss(tf.keras.losses.Loss):
    """Composition of Focal and Huber losses."""

    def __init__(self,
                 num_classes=75,
                 embedding_size = 32,
                 alpha=0.25,
                 gamma=1.5,
                 label_smoothing=0.1,
                 delta=1.0,
                 name='effdet_loss',
                 anchor_generator = AnchorsMobileNet
                ):
        """Initialize Focal and Huber loss.

        Args:
            num_classes: an integer number representing number of
                all possible classes in training dataset.
            alpha: a float number for Focal loss formula.
            gamma: a float number for Focal loss formula.
            label_smoothing: a float number of label smoothing intensity.
            delta: a float number representing a threshold in Huber loss
                for choosing between linear and cubic loss.
        """
        super().__init__(name=name)
        
        self.embed_loss = tf.keras.losses.CosineSimilarity()# tf.keras.losses.MeanSquaredLogarithmicError()#tf.keras.losses.CosineSimilarity()#tf.keras.losses.KLDivergence(reduction=tf.keras.losses.Reduction.AUTO)#tf.keras.losses.MeanAbsoluteError()#tf.keras.losses.MeanSquaredError()


        self.box_loss = NWD(delta=2.0)
        self.det_loss = FocalLoss()
        self.num_classes = num_classes
        self.embedding_size = embedding_size
        self.anchors = anchor_generator()

        self.binary_classification = False

        if self.num_classes == 2 : 
            self.binary_classification = True

        self.cce = tf.keras.losses.CategoricalCrossentropy()

    def box_loss_(self, y_true, y_pred):

        box_labels = y_true[..., :4]
        box_preds = y_pred

        #"""
        box_labels = self.anchors.bbox_transform_inv(box_labels)
        box_labels = to_corners(box_labels) 
        box_preds = self.anchors.bbox_transform_inv(box_preds)
        box_preds = to_corners(box_preds) 
        #"""

        positive_mask = tf.cast(tf.greater(y_true[..., 4], -1), dtype=tf.float32)
        box_loss = self.box_loss(box_labels, box_preds)

        box_loss = tf.where(tf.equal(positive_mask, 1.0), box_loss, 0.0)

        normalizer = tf.reduce_sum(positive_mask, axis=-1)
        box_loss = tf.math.divide_no_nan(tf.reduce_sum(box_loss, axis=-1), normalizer)

        return box_loss

    def embed_loss_(self, y_true, y_pred):

        """
        Args:
        w/ embeddings: 
            y_true : [batch_size, num_anchors, embedding_size+1]
            y_pred : [batch_size, num_anchors, embedding_size]
        """
        y_true_class = y_true[...,0]

        y_true_embedding = y_true[...,1:]

        clf_loss = self.embed_loss(y_true_embedding, y_pred) 

        ignore_mask = tf.cast(tf.equal(y_true_class, -2), dtype=tf.float32)

        ### Positive mask will be considered only from [1..N], so the background will be excluded (not as in other loss)
        positive_mask = tf.cast(tf.greater(y_true_class, 0), dtype=tf.float32)

        clf_loss = tf.where(tf.equal(ignore_mask, 1.0), 0.0, clf_loss)
        normalizer = tf.reduce_sum(positive_mask, axis=-1)
        clf_loss = tf.math.divide_no_nan(tf.reduce_sum(clf_loss, axis=-1), normalizer)

        return clf_loss

    def det_loss_(self, y_true, y_pred):
        """
        Args:
            y_true : [batch_size, num_anchors, 1]
            y_pred : [batch_size, num_anchors, num_classes]
        """
            
        y_true = y_true[...,0]

        y_true_target_det = y_true
        
        if self.binary_classification:
            y_true_target_det = tf.where(tf.greater(y_true, 0), 1.0, y_true)
        
        y_true_target_det = tf.where(tf.equal(y_true, -2), 0.0, y_true_target_det)##
        y_true_target_det = tf.where(tf.equal(y_true, -1), 0.0, y_true_target_det)#########################

        ## important to know that tf.one_hot will return all zeros for non-positive labels!!
        det_labels = tf.one_hot(
            tf.cast(y_true_target_det, dtype=tf.int32),
            depth=self.num_classes,
            dtype=tf.float32
        )

        clf_loss = self.det_loss(det_labels, y_pred) 

        ignore_mask = tf.cast(tf.equal(y_true, -2), dtype=tf.float32)
        positive_mask = tf.cast(tf.greater(y_true, -1), dtype=tf.float32) #


        clf_loss = tf.where(tf.equal(ignore_mask, 1.0), 0.0, clf_loss)
        normalizer = tf.reduce_sum(positive_mask, axis=-1)
        clf_loss = tf.math.divide_no_nan(tf.reduce_sum(clf_loss, axis=-1), normalizer)

        return clf_loss

    def zero_loss_(self, yt, yp):
        return 0.0

    @tf.autograph.experimental.do_not_convert
    def call(self, y_true, y_pred):
        """Calculate Focal and Huber losses for every anchor box.

        Args:
            y_true: a tensor of ground truth values with shape (batch_size, num_anchor_boxes, 5)
                representing anchor box correction and class label.
            y_pred: a tensor of predicted values with
                shape (batch_size, num_anchor_boxes, num_classes).

        Returns:
            loss: a float loss value.
        """

        y_pred = tf.cast(y_pred, dtype=tf.float32)

        box_labels = y_true[..., :4]
        box_preds = y_pred[..., :4]

        cls_labels = tf.cast(y_true[..., 4:], dtype=tf.float32)
        cls_preds = y_pred[..., 4:]

        y_true_argmax = tf.argmax(cls_labels, axis=-1)

        positive_mask = tf.cast(tf.greater(y_true_argmax, 0), dtype=tf.float32)
        ignore_mask = tf.cast(tf.equal(y_true_argmax, 0), dtype=tf.float32)

        clf_loss = self.class_loss(cls_labels, cls_preds)
        box_loss = self.box_loss(box_labels, box_preds)

        clf_loss = tf.where(tf.equal(ignore_mask, 1.0), 0.0, clf_loss)
        box_loss = tf.where(tf.equal(positive_mask, 1.0), box_loss, 0.0)



        normalizer = tf.reduce_sum(positive_mask, axis=-1)
        clf_loss = tf.math.divide_no_nan(tf.reduce_sum(clf_loss, axis=-1), normalizer)
        box_loss = tf.math.divide_no_nan(tf.reduce_sum(box_loss, axis=-1), normalizer)
        loss = clf_loss + box_loss
        return loss
