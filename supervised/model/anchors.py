"""Implementation of anchor boxes generator and encoder of training data."""

import tensorflow as tf


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
def compute_iou(boxes_1, boxes_2):
    """Compute intersection over union.

    Args:
        boxes_1: a tensor with shape (N, 4) representing bounding boxes
            where each box is of the format [x, y, width, height].
        boxes_2: a tensor with shape (M, 4) representing bounding boxes
            where each box is of the format [x, y, width, height].

    Returns:
        IOU matrix with shape (N, M).
    """

    boxes_1_corners = to_corners(boxes_1)
    boxes_2_corners = to_corners(boxes_2)

    left_upper = tf.maximum(boxes_1_corners[..., None, :2], boxes_2_corners[..., :2])
    right_lower = tf.minimum(boxes_1_corners[..., None, 2:], boxes_2_corners[..., 2:])
    diff = tf.maximum(0.0, right_lower - left_upper)
    intersection = diff[..., 0] * diff[..., 1]

    boxes_1_area = boxes_1[..., 2] * boxes_1[..., 3]
    boxes_2_area = boxes_2[..., 2] * boxes_2[..., 3]
    union = boxes_1_area[..., None] + boxes_2_area - intersection

    iou = intersection / union
    return tf.clip_by_value(iou, 0.0, 1.0)
    


class Anchors():
    """Anchor boxes generator."""

    def __init__(self,
                 aspect_ratios=[0.5, 1, 2],
                 scales=[0, 1/3, 2/3]):
        """Initialize anchors generator.

        Args:
            aspect_ratios: a list of floats representing aspect
                ratios of anchor boxes on each feature level.
            scales: a list of floats representing different scales
                of anchor boxes on each feature level.
        """
        self._aspect_ratios = aspect_ratios
        self._scales = [2**i for i in scales]
        self._num_anchors = len(aspect_ratios) * len(scales)

        self._strides = [2**i for i in range(3, 8)]
        self._areas = [i**2 for i in [32.0, 64.0, 128.0, 256.0, 512.0]]
        self._anchor_dims = self._compute_dims()

    def _compute_dims(self):
        """Compute height and width for each anchor box on each level.

        Returns:
            A float tensor with shape (5, num_anchors, 2) where each
                pair representing height and width of anchor box.
        """
        all_dims = list()
        for area in self._areas:
            level_dims = list()
            for aspect_ratio in self._aspect_ratios:
                height = tf.math.sqrt(area * aspect_ratio)
                width = area / height
                dims = tf.cast([height, width], tf.float32)
                for scale in self._scales:
                    level_dims.append(dims * scale)
            all_dims.append(tf.stack(level_dims, axis=0))
        return tf.stack(all_dims, axis=0)

    @tf.function
    def _get_anchors(self, feature_height, feature_width, level):
        """Get anchors for with given height and width on given level.

        Args:
            feature_height: an integer representing height of feature map.
                Should be divisible by 2**level.
            feature_width: an integer representing width of feature map.
                Should be divisible by 2**level.
            level: an integer from range [3, 7] representing level
                of feature map.
        """
        rx = tf.range(feature_width, dtype=tf.float32) + .5
        ry = tf.range(feature_height, dtype=tf.float32) + .5
        xs = tf.tile(tf.reshape(rx, [1, -1]), [tf.shape(ry)[0], 1])
        ys = tf.tile(tf.reshape(ry, [-1, 1]), [1, tf.shape(rx)[0]])

        centers = tf.stack([xs, ys], axis=-1) * self._strides[level - 3]
        centers = tf.reshape(centers, [-1, 1, 2])
        centers = tf.tile(centers, [1, self._num_anchors, 1])
        centers = tf.reshape(centers, [-1, 2])

        dims = tf.tile(self._anchor_dims[level - 3], [feature_height * feature_width, 1])
        return tf.concat([centers, dims], axis=-1)

    def get_anchors(self, image_height, image_width):
        """Get anchors for given height and width on all levels.

        Args:
            image_height: an integer representing height of image.
            image_width: an integer representing width of image.
        """
        anchors = [
            self._get_anchors(
                tf.math.ceil(image_height / 2**i),
                tf.math.ceil(image_width / 2**i),
                i
            ) for i in range(3, 8)
        ]
        return tf.concat(anchors, axis=0)


class BaseAnchor():

    def __init__(self):
        self._anchors = []
        self._scale_factors = []
        self._deltas = []

    def get_anchors(self, width=None, height=None):
        raise NotImplementedError 

    def approx_ex(self, x):
        return 1 + x + (x*x)/2 + (x*x*x)/6 + (x*x*x*x)/24 + (x*x*x*x*x)/120 + (x*x*x*x*x*x)/720

    def bbox_transform_inv(self, boxes, anchor_boxes=None, approx=False):

        if anchor_boxes is None:
            anchor_boxes = self._anchors
        
        boxes = tf.cast(boxes , tf.float32) * [self._deltas]

        if approx:

            x2 = boxes[..., 2:]*boxes[..., 2:]
            wh = (1+boxes[..., 2:]+x2/2 + (x2*boxes[...,2:])/6)

            post_box = tf.concat([
                boxes[..., :2] * self._anchors[..., 2:] + self._anchors[..., :2],
                self.approx_ex(boxes[..., 2:])*self._anchors[...,2:]
            ], axis=-1)  * [self._scale_factors]
        else:
            post_box = tf.concat([
                boxes[..., :2] * self._anchors[..., 2:] + self._anchors[..., :2],
                tf.exp(boxes[..., 2:])*self._anchors[...,2:]
            ], axis=-1)  * [self._scale_factors]

        return post_box

class AnchorsMobileNet(BaseAnchor):

    def __init__(self, deltas=[0.1, 0.1, 0.2, 0.2], scale_factors=[128,128,128,128]):

        super().__init__()
        self._scale_factors = scale_factors
        self._deltas = deltas
        self._anchors = self._get_anchors()

    def get_anchors(self, width=None, height=None):
        return self._anchors

    def _get_anchors(self,width=128, height=128):
        # target 341 anchors to be generated 

        """
        Returns an anchor of shape: [288, 4], last dimension having the format of [x,y,w,h]

        """
        target_anchor_num = 1024

        height_step = height // (target_anchor_num // 32) ## 32 is from (96-32)/2

        additional_anchors = target_anchor_num % 32

        anchors = []
        
        for i in range(0,height,height_step):
            for j in range(32, 96, 2): # the middle 64 points
                anchors.append([j, i+16, 5,32])

        for i in range(additional_anchors):
            anchors.append([30+2*i,3*i+16, 4, 25])
        """
        for i in range(0,21):
            anchors.append([30+3*i,6*i+16, 4, 25])
        
        for i in range(0,43):
            anchors.append([96-4*i,6*i+20, 4, 25])
        #"""
        anchors = tf.cast(tf.stack(anchors, axis=0), tf.float32) / self._scale_factors
        


        return anchors


class SamplesEncoder():
    """Enchoder of training batches."""

    def __init__(self,
                 aspect_ratios=[0.5, 1, 2],
                 scales=[0, 1/3, 2/3],
                 anchor_generator=AnchorsMobileNet):
        #self._anchors = Anchors(aspect_ratios=aspect_ratios, scales=scales)

        self._anchors = anchor_generator() #SpikeAnchors()
        self._box_variance = tf.cast(
            [0.1, 0.1, 0.2, 0.2], tf.float32
        )

    def _match_anchor_boxes(self, anchor_boxes, gt_boxes, match_iou=0.3, ignore_iou=0.1): # 0.3 0.1
        """Assign ground truth boxes to all anchor boxes."""

        iou = compute_iou(anchor_boxes, gt_boxes)


        max_iou = tf.reduce_max(iou, axis=1)
        matched_gt_idx = tf.argmax(iou, axis=1)
        positive_mask = tf.greater_equal(max_iou, match_iou)
        negative_mask = tf.less(max_iou, ignore_iou)
        ignore_mask = tf.logical_not(tf.logical_or(positive_mask, negative_mask))
        return (
            matched_gt_idx,
            tf.cast(positive_mask, dtype=tf.float32),
            tf.cast(ignore_mask, dtype=tf.float32),
        )

    @tf.autograph.experimental.do_not_convert
    def _compute_box_target(self, anchor_boxes, matched_gt_boxes):
        box_target = tf.concat(
            [
                (matched_gt_boxes[:, :2] - anchor_boxes[:, :2]) / anchor_boxes[:, 2:],
                tf.math.log(matched_gt_boxes[:, 2:] / anchor_boxes[:, 2:]),
            ],
            axis=-1,
        )
        box_target = box_target / self._box_variance
        return box_target

    def _encode_sample(self, image_shape, gt_boxes, classes):

                
        """
        Classes should be the following: 
        0 - for negative samples 
        > 0 - for positive samples 
        """

        classes = tf.cast(classes, dtype=tf.float32)
        
        anchor_boxes = self._anchors.get_anchors(image_shape[1], image_shape[2])

        matched_gt_idx, positive_mask, ignore_mask = self._match_anchor_boxes(
            anchor_boxes, gt_boxes
        )

        matched_gt_boxes = tf.gather(gt_boxes, matched_gt_idx)
        matched_gt_classes = tf.gather(classes, matched_gt_idx)

        box_target = self._compute_box_target(anchor_boxes, matched_gt_boxes)

        class_target = tf.where(tf.equal(positive_mask, 1.0), matched_gt_classes, -1.0)
        class_target = tf.where(tf.equal(ignore_mask, 1.0), -2.0, class_target)
        class_target = tf.expand_dims(class_target, axis=-1)

        return box_target, class_target
        
        label = tf.concat([box_target, class_target], axis=-1)

        return label

    def encode_batch(self, images, gt_boxes, classes):
        """Encode batch for training."""

        images_shape = tf.shape(images)
        batch_size = images_shape[0]

        labels = tf.TensorArray(dtype=tf.float32, size=batch_size)
        for i in range(batch_size):
            label = self._encode_sample(images_shape, gt_boxes[i], classes[i])
            labels = labels.write(i, label)
        #images = tf.keras.applications.efficientnet.preprocess_input(images)
        return images, labels.stack()
