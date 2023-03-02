"""Implementation of utility functions."""

import tensorflow as tf
import numpy as np

@tf.autograph.experimental.do_not_convert
def to_xywh(bbox):
    """Convert [x_min, y_min, x_max, y_max] to [x, y, width, height]."""
    return tf.concat(
        [(bbox[..., :2] + bbox[..., 2:]) / 2.0, (bbox[..., 2:] - bbox[..., :2])], axis=-1
    )


@tf.autograph.experimental.do_not_convert
def to_corners(bbox):
    """Convert [x, y, width, height] to [x_min, y_min, x_max, y_max]."""
    return tf.concat(
        [bbox[..., :2] - bbox[..., 2:] / 2.0, bbox[..., :2] + bbox[..., 2:] / 2.0], axis=-1
    )


@tf.autograph.experimental.do_not_convert
def compute_iou(boxes_1, boxes_2, corner_format=False):
    """Compute intersection over union.

    Args:
        boxes_1: a tensor with shape (N, 4) representing bounding boxes
            where each box is of the format [x, y, width, height].
        boxes_2: a tensor with shape (M, 4) representing bounding boxes
            where each box is of the format [x, y, width, height].

    Returns:
        IOU matrix with shape (N, M).
    """

    if not corner_format: 
        boxes_1_corners = to_corners(boxes_1)
        boxes_2_corners = to_corners(boxes_2)
    else:
        boxes_1_corners = boxes_1
        boxes_2_corners = boxes_2

        boxes_1 = to_xywh(boxes_1)
        boxes_2 = to_xywh(boxes_2)

    left_upper = tf.maximum(boxes_1_corners[..., None, :2], boxes_2_corners[..., :2])
    right_lower = tf.minimum(boxes_1_corners[..., None, 2:], boxes_2_corners[..., 2:])
    diff = tf.maximum(0.0, right_lower - left_upper)
    intersection = diff[..., 0] * diff[..., 1]

    boxes_1_area = boxes_1[..., 2] * boxes_1[..., 3]
    boxes_2_area = boxes_2[..., 2] * boxes_2[..., 3]
    union = boxes_1_area[..., None] + boxes_2_area - intersection

    iou = intersection / union
    return tf.clip_by_value(iou, 0.0, 1.0)

@tf.autograph.experimental.do_not_convert
def compute_overlap(gt_boxes, pred_box):
    """Compute intersection over union.

    Args:
        boxes_1_corners: a tensor with shape (1, 4) representing bounding boxes
            where each box is of the format [x1, y2, x2, y2]. This is the GT!!
        boxes_2_corners: a tensor with shape (N, 4) representing bounding boxes
            where each box is of the format [x1, y2, x2, y2].

    Returns:
        IOU matrix with shape (N, ).
    """
    results = []

    for i in range(gt_boxes.shape[0]):
        boxes_1_corners = gt_boxes[i:i+1, :4]
        boxes_2_corners = pred_box[...,:4]

        #print("Box 1")
        #print(boxes_1_corners)
        #print("Box 2")
        #print(boxes_2_corners)

        left_upper = np.maximum(boxes_1_corners[..., :2], boxes_2_corners[..., :2])
        right_lower = np.minimum(boxes_1_corners[..., 2:], boxes_2_corners[..., 2:])
        diff = np.maximum(0.0, right_lower - left_upper)
        intersection = diff[..., 0] * diff[..., 1]
        boxes_1_area = (boxes_1_corners[..., 2] - boxes_1_corners[..., 0]) * (boxes_2_corners[..., 3] - boxes_2_corners[..., 1])

        overlap = intersection / boxes_1_area

        results.append(np.clip(overlap, 0.0, 1.0))  

    results = np.squeeze(results, axis=-1)
    return results

def random_horizontal_flip(image, boxes):
    """Flip image and boxes horizontally."""

    if tf.random.uniform(()) >= 0.5:
        image = tf.image.flip_left_right(image)
        boxes = tf.stack(
            [1 - boxes[..., 2], boxes[..., 1], 1 - boxes[..., 0], boxes[..., 3]], axis=-1
        )
    return image, boxes


def resize_and_pad(image, target_side=512.0, max_side=1024.0, scale_jitter=[0.1, 2.0], stride=128.0):
    """Resize image, apply scale jittering and pad with zeros to make image divisible by stride."""

    image_shape = tf.cast(tf.shape(image)[:2], tf.float32)
    bigger_side = tf.reduce_max(image_shape)
    target_side = target_side
    if scale_jitter:
        target_side = bigger_side * tf.random.uniform((), scale_jitter[0], scale_jitter[1], dtype=tf.float32)
    scale_coeff = target_side / bigger_side
    if target_side > max_side:
        scale_coeff = max_side / bigger_side

    new_image_shape = image_shape * scale_coeff
    new_image = tf.image.resize(image, tf.cast(new_image_shape, tf.int32))
    padded_image_shape = tf.cast(tf.math.ceil(new_image_shape / stride) * stride, dtype=tf.int32)
    padded_image = tf.image.pad_to_bounding_box(
        new_image, 0, 0, padded_image_shape[0], padded_image_shape[1]
    )

    return padded_image, new_image_shape, scale_coeff

def filter_detections(
        boxes,
        classification,
        alphas=None,
        ratios=None,
        class_specific_filter=True,
        nms=True,
        score_threshold=0.01,
        max_detections=100,
        nms_threshold=0.5,
        detect_quadrangle=False,
):
    """
    Filter detections using the boxes and classification values.
    Args
        boxes: Tensor of shape (num_boxes, 4) containing the boxes in (x1, y1, x2, y2) format.
        classification: Tensor of shape (num_boxes, num_classes) containing the classification scores.
        other: List of tensors of shape (num_boxes, ...) to filter along with the boxes and classification scores.
        class_specific_filter: Whether to perform filtering per class, or take the best scoring class and filter those.
        nms: Flag to enable/disable non maximum suppression.
        score_threshold: Threshold used to prefilter the boxes with.
        max_detections: Maximum number of detections to keep.
        nms_threshold: Threshold for the IoU value to determine when a box should be suppressed.
    Returns
        A list of [boxes, scores, labels, other[0], other[1], ...].
        boxes is shaped (max_detections, 4) and contains the (x1, y1, x2, y2) of the non-suppressed boxes.
        scores is shaped (max_detections,) and contains the scores of the predicted class.
        labels is shaped (max_detections,) and contains the predicted label.
        other[i] is shaped (max_detections, ...) and contains the filtered other[i] data.
        In case there are less than max_detections detections, the tensors are padded with -1's.
    """

    def _filter_detections(scores_, labels_):
        # threshold based on score
        # (num_score_keeps, 1)


        indices_ = tf.where(tf.keras.backend.greater(scores_, score_threshold))

        if nms:
            # (num_score_keeps, 4)
            filtered_boxes = tf.gather_nd(boxes, indices_)
            # In [4]: scores = np.array([0.1, 0.5, 0.4, 0.2, 0.7, 0.2])
            # In [5]: tf.greater(scores, 0.4)
            # Out[5]: <tf.Tensor: id=2, shape=(6,), dtype=bool, numpy=array([False,  True, False, False,  True, False])>
            # In [6]: tf.where(tf.greater(scores, 0.4))
            # Out[6]:
            # <tf.Tensor: id=7, shape=(2, 1), dtype=int64, numpy=
            # array([[1],
            #        [4]])>
            #
            # In [7]: tf.gather(scores, tf.where(tf.greater(scores, 0.4)))
            # Out[7]:
            # <tf.Tensor: id=15, shape=(2, 1), dtype=float64, numpy=
            # array([[0.5],
            #        [0.7]])>
            filtered_scores = tf.keras.backend.gather(scores_, indices_)[:, 0]

            # perform NMS
            # filtered_boxes = tf.concat([filtered_boxes[..., 1:2], filtered_boxes[..., 0:1],
            #                             filtered_boxes[..., 3:4], filtered_boxes[..., 2:3]], axis=-1)
            nms_indices = tf.image.non_max_suppression(filtered_boxes, filtered_scores, max_output_size=max_detections,
                                                       iou_threshold=nms_threshold)

            # filter indices based on NMS
            # (num_score_nms_keeps, 1)
            indices_ = tf.keras.backend.gather(indices_, nms_indices)

        # add indices to list of all indices
        # (num_score_nms_keeps, )
        labels_ = tf.gather_nd(labels_, indices_)
        # (num_score_nms_keeps, 2)
        indices_ = tf.keras.backend.stack([indices_[:, 0], labels_], axis=1)

        return indices_

    if class_specific_filter:
        all_indices = []
        # perform per class filtering
        for c in range(int(classification.shape[1])):
            scores = classification[:, c]
            labels = c * tf.ones((tf.keras.backend.shape(scores)[0],), dtype='int64')
            all_indices.append(_filter_detections(scores, labels))

        # concatenate indices to single tensor
        # (concatenated_num_score_nms_keeps, 2)
        indices = tf.keras.backend.concatenate(all_indices, axis=0)
    else:
        scores = tf.keras.backend.max(classification, axis=1)
        labels = tf.keras.backend.argmax(classification, axis=1)
        indices = _filter_detections(scores, labels)

    # select top k
    scores = tf.gather_nd(classification, indices)
    labels = indices[:, 1]
    scores, top_indices = tf.nn.top_k(scores, k=tf.keras.backend.minimum(max_detections, tf.keras.backend.shape(scores)[0]))

    # filter input using the final set of indices
    indices = tf.keras.backend.gather(indices[:, 0], top_indices)
    boxes = tf.keras.backend.gather(boxes, indices)
    labels = tf.keras.backend.gather(labels, top_indices)

    # zero pad the outputs
    pad_size = tf.keras.backend.maximum(0, max_detections - tf.keras.backend.shape(scores)[0])
    boxes = tf.pad(boxes, [[0, pad_size], [0, 0]], constant_values=-1)
    scores = tf.pad(scores, [[0, pad_size]], constant_values=-1)
    labels = tf.pad(labels, [[0, pad_size]], constant_values=-1)
    labels = tf.keras.backend.cast(labels, 'int32')

    # set shapes, since we know what they are
    boxes.set_shape([max_detections, 4])
    scores.set_shape([max_detections])
    labels.set_shape([max_detections])

    if detect_quadrangle:
        alphas = tf.keras.backend.gather(alphas, indices)
        ratios = tf.keras.backend.gather(ratios, indices)
        alphas = tf.pad(alphas, [[0, pad_size], [0, 0]], constant_values=-1)
        ratios = tf.pad(ratios, [[0, pad_size]], constant_values=-1)
        alphas.set_shape([max_detections, 4])
        ratios.set_shape([max_detections])
        return [boxes, scores, alphas, ratios, labels]
    else:
        return [boxes, scores, labels]

from .anchors import SpikeAnchors, AnchorsMobileNet
class FilterDetections(tf.keras.layers.Layer):
    """
    Keras layer for filtering detections using score threshold and NMS.
    """

    def __init__(
            self,
            nms=True,
            class_specific_filter=True,
            nms_threshold=0.5,
            score_threshold=0.01,
            max_detections=100,
            parallel_iterations=32,
            detect_quadrangle=False,
            **kwargs
    ):
        """
        Filters detections using score threshold, NMS and selecting the top-k detections.
        Args
            nms: Flag to enable/disable NMS.
            class_specific_filter: Whether to perform filtering per class, or take the best scoring class and filter those.
            nms_threshold: Threshold for the IoU value to determine when a box should be suppressed.
            score_threshold: Threshold used to prefilter the boxes with.
            max_detections: Maximum number of detections to keep.
            parallel_iterations: Number of batch items to process in parallel.
        """
        self.nms = nms
        self.class_specific_filter = class_specific_filter
        self.nms_threshold = nms_threshold
        self.score_threshold = score_threshold
        self.max_detections = max_detections
        self.parallel_iterations = parallel_iterations
        self.detect_quadrangle = detect_quadrangle

        self.spike_anchors = SpikeAnchors()
        super(FilterDetections, self).__init__(**kwargs)

    def call(self, boxes, classification, **kwargs):
        """
        Constructs the NMS graph.
        Args
            inputs : List of [boxes, classification, other[0], other[1], ...] tensors.
        """

        #boxes = inputs[0]
        #classification = inputs[1]

        #boxes = self.spike_anchors.bbox_transform_inv(boxes)

        if self.detect_quadrangle:
            alphas = inputs[2]
            ratios = inputs[3]

        # wrap nms with our parameters
        def _filter_detections(args):
            boxes_ = args[0]
            classification_ = args[1]
            alphas_ = args[2] if self.detect_quadrangle else None
            ratios_ = args[3] if self.detect_quadrangle else None

            return filter_detections(
                boxes_,
                classification_,
                alphas_,
                ratios_,
                nms=self.nms,
                class_specific_filter=self.class_specific_filter,
                score_threshold=self.score_threshold,
                max_detections=self.max_detections,
                nms_threshold=self.nms_threshold,
                detect_quadrangle=self.detect_quadrangle,
            )

        # call filter_detections on each batch item
        if self.detect_quadrangle:
            outputs = tf.map_fn(
                _filter_detections,
                elems=[boxes, classification, alphas, ratios],
                dtype=['float32', 'float32', 'float32', 'float32', 'int32'],
                parallel_iterations=self.parallel_iterations
            )
        else:
            outputs = tf.map_fn(
                _filter_detections,
                elems=[boxes, classification],
                dtype=['float32', 'float32', 'int32'],
                parallel_iterations=self.parallel_iterations
            )

        return outputs

    def compute_output_shape(self, input_shape):
        """
        Computes the output shapes given the input shapes.
        Args
            input_shape : List of input shapes [boxes, classification].
        Returns
            List of tuples representing the output shapes:
            [filtered_boxes.shape, filtered_scores.shape, filtered_labels.shape, filtered_other[0].shape, filtered_other[1].shape, ...]
        """
        if self.detect_quadrangle:
            return [
                (input_shape[0][0], self.max_detections, 4),
                (input_shape[1][0], self.max_detections),
                (input_shape[1][0], self.max_detections, 4),
                (input_shape[1][0], self.max_detections),
                (input_shape[1][0], self.max_detections),
            ]
        else:
            return [
                (input_shape[0][0], self.max_detections, 4),
                (input_shape[1][0], self.max_detections),
                (input_shape[1][0], self.max_detections),
            ]

    def compute_mask(self, inputs, mask=None):
        """
        This is required in Keras when there is more than 1 output.
        """
        return (len(inputs) + 1) * [None]

    def get_config(self):
        """
        Gets the configuration of this layer.
        Returns
            Dictionary containing the parameters of this layer.
        """
        config = super(FilterDetections, self).get_config()
        config.update({
            'nms': self.nms,
            'class_specific_filter': self.class_specific_filter,
            'nms_threshold': self.nms_threshold,
            'score_threshold': self.score_threshold,
            'max_detections': self.max_detections,
            'parallel_iterations': self.parallel_iterations,
        })

        return config

def post_process_anchor(
    inputs, 
    threshold=0.5, 
    num_classes=75, 
    score_threshold = 0.5, 
    class_specific_filter=False, 
    min_spike_length=15, 
    max_spike_length = 64, 
    max_detection=10,
    anchor_generator = SpikeAnchors
    ):
    
    """
    Args:
        inputs: [boxes, embedding, classes]
        
                    boxes: Tensor of shape (batch_size, num_boxes, 4) containing the boxes in (x1, y1, w, h) format.
                    embedding: Tensor of shape (batch_size, num_boxes, embedding_size)
                    classes: Tensor of shape (batch_size, num_boxes, num_classes) 
    Returns:
        (boxes, scores, labels) : Tuple 
            boxes - with shape  [n, 4] in (x1,y1,x2,y2) format
            scores - with shape [n, ]
            labels - with shape [n, ]
            n (default) = 100
    """
    boxes = inputs[0]
    embedding = inputs[1]
    classes = inputs[2] #inputs[1]

    boxes = boxes[..., :4]


    anchors = anchor_generator()

    boxes = anchors.bbox_transform_inv(boxes)

    boxes = to_corners(boxes)

    # Size condition threshold is implemented to ensure that no FP detections are made in the middle of the sample

    # Filtering by height
    size_condition = tf.logical_and(tf.greater(boxes[...,1], 1), tf.less(boxes[...,3], anchors._scale_factors[0]-1))
    size_condition = tf.logical_and(size_condition, tf.less(boxes[...,3]-boxes[...,1], min_spike_length))
    size_condition = tf.logical_or(size_condition, tf.greater(boxes[...,3]-boxes[...,1], max_spike_length))
    

    # Filtering by width 
    size_condition = tf.logical_or(
        size_condition, 
        tf.logical_and(
            tf.logical_and(
                tf.greater(boxes[...,0], 32+2), 
                tf.less(boxes[...,2], 96-2)
                ), 
            tf.less(boxes[...,2]-boxes[...,0], 3)
            )
        )

    classes = tf.where(tf.expand_dims(size_condition, axis=-1), -1.0, classes)


    classes = tf.where(tf.equal(classes, 0), -1.0, classes)

    classes = tf.cast(classes, tf.float32) # just to be sure that every value is float 

    #boxes = tf.clip_by_value(boxes, clip_value_min=0, clip_value_max=anchors._scale_factors[0])

    #"""
    filter_layer = FilterDetections(
        nms=True,
        class_specific_filter = class_specific_filter,
        nms_threshold = threshold,
        score_threshold=score_threshold,
        max_detections=max_detection,
    )  
    boxes, scores, labels = filter_layer(boxes, classes)


    boxes = tf.clip_by_value(boxes, clip_value_min=0, clip_value_max=anchors._scale_factors[0])

    mask = tf.not_equal(labels, 0)


    scores = tf.where(mask, scores, 0)
    labels = tf.where(mask, labels, -1)
    

    return tf.cast(boxes, tf.float32), tf.cast(scores, tf.float32), tf.cast(labels, tf.float32)



import matplotlib.pyplot as plt 
from matplotlib import patches
def show_patches(image, bboxes, bboxes_2=[], scores=[], labels=[], w_x=1, h_y=1, to_corner=False, score_threshold=None):

    if isinstance(bboxes, list):
        bboxes = bboxes[0] # in case a direct input is given from the post process function

    assert (len(bboxes.shape) ==3 and len(image.shape)==4 and bboxes.shape[0] == image.shape[0] ) or (len(bboxes.shape) == 2 and len(image.shape) == 3), f"Wrong shapes: images {image.shape} and bboxes {bboxes.shape}"
    
    if len(bboxes.shape) == 2:
        image = tf.expand_dims(image, axis=0)
        bboxes = tf.expand_dims(bboxes, axis=0)

    if to_corner:
        #bboxes = bboxes * [128,128,128,128] ### HARDCODED 
        bboxes = to_corners(bboxes)


    def add_patch(ax, box, label, color="red", lw=2):

        if label < 1:
            return

        x = box[0]
        y = box[1]
        w = box[2] - x
        h = box[3] - y

        ax.add_patch(patches.Rectangle((x,y),w,h, fill=False, edgecolor=color, lw=lw))


    if len(bboxes_2) > 0:
        fig, ax = plt.subplots(figsize = (6,9))
        ax.xaxis.tick_top()

    for k, (im, boxes) in enumerate(zip(image, bboxes)):


        if len(bboxes_2) == 0:
            fig, ax = plt.subplots(figsize = (6,9))
            ax.xaxis.tick_top()

        if len(labels) > k+1:
            print(labels[k])

        ax.imshow((im - tf.math.reduce_min(im)) / (tf.math.reduce_max(im)-tf.math.reduce_min(im)))

        if (len(labels) ==0):
            for i, box in enumerate(boxes):

                if score_threshold is not None and scores[k][i] < score_threshold:
                    continue

                add_patch(ax, box, label=1)

        else:
            for i, (label, box) in enumerate(zip(labels[k], boxes)):

                if score_threshold is not None and scores[k][i] < score_threshold:
                    continue

                add_patch(ax, box, label)

        if len(bboxes_2) == 0:
            plt.show()


    if len(bboxes_2) > 0:
        for box in bboxes_2:
                
            add_patch(ax, box, 1, color="yellow", lw=1)

        plt.show()




