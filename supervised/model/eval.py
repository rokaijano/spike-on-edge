import numpy as np
import tensorflow as tf
from model.utils import post_process_anchor, compute_overlap

from model.anchors import SpikeAnchors

def _compute_ap(recall, precision):
    """
    Compute the average precision, given the recall and precision curves.
    Code originally from https://github.com/rbgirshick/py-faster-rcnn.
    Args:
        recall: The recall curve (list).
        precision: The precision curve (list).
    Returns:
        The average precision as computed in py-faster-rcnn.
    """
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.], recall, [1.]))
    mpre = np.concatenate(([0.], precision, [0.]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap

def _get_detections_and_annotation(generator, model, num_classes=75, score_threshold=0.5):
    """
    Get the detections from the model using the generator.
    The result is a list of lists such that the size is:
        all_detections[num_images][num_classes] = detections[num_class_detections, 5]
    Args:
        generator: The generator used to run images through the model.
        model: The model to run on the images.
        score_threshold: The score confidence threshold to use.
        max_detections: The maximum number of detections to use per image.
        save_path: The path to save the images with visualized detections to.
    Returns:
        A list of lists containing the detections for each image in the generator.
    """
    all_detections = []
    all_annotations = []

                      
    for batch in generator:

        image = batch[0]
        gt = batch[1]
        # run network

        
        image_boxes, image_scores, image_labels = post_process_anchor(model(image), score_threshold = score_threshold)
        bboxes_gt, _, cls_id = post_process_anchor(gt)
        
        """
        images_boxes: [batch_size, max_boxes, 4]
        image_scores: [batch_size, max_boxes, ]
        image_scores: [batch_size, max_boxes, ]
        """
        """
        bboxes_gt = gt[0]
        bboxes_gt = bboxes_gt[..., :4]
        bboxes_gt = SpikeAnchors().bbox_transform_inv(bboxes_gt)

        cls_id = gt[1] 
        cls_id = tf.where(tf.greater_equal(cls_id, 0.0), cls_id, 0.0)
        """

        #print(bboxes_gt)
        cls_id = tf.expand_dims(cls_id, axis=-1)

        detections =  tf.concat([image_boxes, np.expand_dims(image_scores, axis=-1), np.expand_dims(image_labels, axis=-1)], axis=-1).numpy() # (batch_size, max_boxes, 6)
        annotations = tf.concat([bboxes_gt, cls_id], axis=-1).numpy() # (batch_size, 341, 5)

        assert annotations.shape[0] == detections.shape[0], f"Annotation and detection shape mismatch: {annotations.shape} and {detections.shape}"

        for k in range(annotations.shape[0]):
            all_detections.append([])
            all_annotations.append([])

            for class_id in range(1,num_classes):
                all_detections[-1].append(detections[k, detections[k, :, -1] == class_id, :-1].copy())
                all_annotations[-1].append(annotations[k, annotations[k, :,-1] == class_id, :].copy()) # .copy()

            

    return all_detections, all_annotations
    



def evaluate(
        generator,
        model,
        iou_threshold=0.5,
        num_classes = 75,
        score_threshold = 0.5,
        epoch=0
):
    """
    Evaluate a given dataset using a given model.
    Args:
        generator: The generator that represents the dataset to evaluate.
        model: The model to evaluate.
        iou_threshold: The threshold used to consider when a detection is positive or negative.
        score_threshold: The score confidence threshold to use for detections.
        max_detections: The maximum number of detections to use per image.
        visualize: Show the visualized detections or not.
    Returns:
        A dict mapping class names to mAP scores.
    """
    # gather all detections and annotations
    all_detections, all_annotations = _get_detections_and_annotation(generator, model, num_classes=num_classes, score_threshold=score_threshold)

    average_precisions = {}
    accuracy = [] # spikeforest type accuracy 
    num_tp = 0
    num_fp = 0

    # process detections and annotations
    for label in range(num_classes-1):

        print(f"Processing label {label}/{num_classes}", end='\r')

        false_positives = np.zeros((0,))
        true_positives = np.zeros((0,))
        scores = np.zeros((0,))
        num_annotations = 0.0

        for i in range(len(all_detections)):
            detections = all_detections[i][label]
            annotations = all_annotations[i][label]
            num_annotations += annotations.shape[0]
            detected_annotations = []

            for d in detections:
                scores = np.append(scores, d[4])

                if annotations.shape[0] == 0:
                    false_positives = np.append(false_positives, 1)
                    true_positives = np.append(true_positives, 0)
                    continue

                d = np.expand_dims(d, axis=0)

                overlaps = compute_overlap(annotations, d)
                assigned_annotation = np.argmax(overlaps)


                max_overlap = overlaps[assigned_annotation]

                if max_overlap >= iou_threshold and assigned_annotation not in detected_annotations:
                    false_positives = np.append(false_positives, 0)
                    true_positives = np.append(true_positives, 1)
                    detected_annotations.append(assigned_annotation)
                else:
                    false_positives = np.append(false_positives, 1)
                    true_positives = np.append(true_positives, 0)

        # no annotations -> AP for this class is 0 (is this correct?)
        if num_annotations == 0:
            average_precisions[label] = 0, 0
            continue

        # sort by score
        indices = np.argsort(-scores)
        false_positives = false_positives[indices]
        true_positives = true_positives[indices]

        # compute false positives and true positives
        false_positives = np.cumsum(false_positives)
        true_positives = np.cumsum(true_positives)

        if false_positives.shape[0] == 0:
            num_fp += 0
        else:
            num_fp += false_positives[-1]
        if true_positives.shape[0] == 0:
            num_tp += 0
        else:
            num_tp += true_positives[-1]

        # compute recall and precision
        recall = true_positives / num_annotations
        precision = true_positives / np.maximum(true_positives + false_positives, np.finfo(np.float64).eps)
        false_negatives = num_annotations - true_positives

        if len(true_positives) == 0:
            acc = 0
        else:
            acc = np.max(true_positives) / np.max(true_positives + false_positives + false_negatives)

        accuracy.append(acc)

        # compute average precision
        average_precision = _compute_ap(recall, precision)
        average_precisions[label] = average_precision, num_annotations

    print('num_fp={}, num_tp={}'.format(num_fp, num_tp))
    print(f"mAP = {np.mean(average_precision)}")
    print(f"Average accuracy = {np.mean(accuracy)}")

    return average_precisions


