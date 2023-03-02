import tensorflow as tf
import numpy as np
from model.utils import *
from typing import List, Tuple
T = tf.Tensor

def tflite_nms_implements_signature(params):
	"""`experimental_implements` signature for TFLite's custom NMS op.
	This signature encodes the arguments to correctly initialize TFLite's custom
	post-processing op in the MLIR converter.
	For details on `experimental_implements` see here:
	https://www.tensorflow.org/api_docs/python/tf/function
	Args:
	params: a dict of parameters.
	Returns:
	String encoding of a map from attribute keys to values.
	"""
	scale_value = 128.0
	nms_configs = params['nms_configs']
	iou_thresh = nms_configs['iou_thresh'] or 0.5
	score_thresh = nms_configs['score_thresh'] or float('-inf')
	max_detections = params['tflite_max_detections']

	TFLITE_MAX_CLASSES_PER_DETECTION = 1
	TFLITE_DETECTION_POSTPROCESS_FUNC = 'TFLite_Detection_PostProcess'
	TFLITE_USE_REGULAR_NMS = False

	implements_signature = [
	  'name: "%s"' % TFLITE_DETECTION_POSTPROCESS_FUNC,
	  'attr { key: "max_detections" value { i: %d } }' % max_detections,
	  'attr { key: "max_classes_per_detection" value { i: %d } }' %
	  TFLITE_MAX_CLASSES_PER_DETECTION,
	  'attr { key: "use_regular_nms" value { b: %s } }' %
	  str(TFLITE_USE_REGULAR_NMS).lower(),
	  'attr { key: "nms_score_threshold" value { f: %f } }' % score_thresh,
	  'attr { key: "nms_iou_threshold" value { f: %f } }' % iou_thresh,
	  'attr { key: "y_scale" value { f: %f } }' % scale_value,
	  'attr { key: "x_scale" value { f: %f } }' % scale_value,
	  'attr { key: "h_scale" value { f: %f } }' % scale_value,
	  'attr { key: "w_scale" value { f: %f } }' % scale_value,
	  'attr { key: "num_classes" value { i: %d } }' % params['num_classes']
	]
	implements_signature = ' '.join(implements_signature)

	return implements_signature

def tflite_nms2_implements_signature(params):

	## This is the version where the indices are given back rather being prefiltered
	"""`experimental_implements` signature for TFLite's custom NMS op.
	This signature encodes the arguments to correctly initialize TFLite's custom
	post-processing op in the MLIR converter.
	For details on `experimental_implements` see here:
	https://www.tensorflow.org/api_docs/python/tf/function
	Args:
	params: a dict of parameters.
	Returns:
	String encoding of a map from attribute keys to values.
	"""
	scale_value = 128.0
	nms_configs = params['nms_configs']
	iou_thresh = nms_configs['iou_thresh'] or 0.5
	score_thresh = nms_configs['score_thresh'] or float('-inf')
	max_detections = params['tflite_max_detections']

	TFLITE_MAX_CLASSES_PER_DETECTION = 1
	TFLITE_DETECTION_POSTPROCESS_FUNC = 'NON_MAX_SUPPRESSION_V5'
	TFLITE_USE_REGULAR_NMS = False

	implements_signature = [
	  'name: "%s"' % TFLITE_DETECTION_POSTPROCESS_FUNC,
	  'attr { key: "max_detections" value { i: %d } }' % max_detections,
	  'attr { key: "max_classes_per_detection" value { i: %d } }' %
	  TFLITE_MAX_CLASSES_PER_DETECTION,
	  'attr { key: "use_regular_nms" value { b: %s } }' %
	  str(TFLITE_USE_REGULAR_NMS).lower(),
	  'attr { key: "nms_score_threshold" value { f: %f } }' % score_thresh,
	  'attr { key: "nms_iou_threshold" value { f: %f } }' % iou_thresh,
	  'attr { key: "y_scale" value { f: %f } }' % scale_value,
	  'attr { key: "x_scale" value { f: %f } }' % scale_value,
	  'attr { key: "h_scale" value { f: %f } }' % scale_value,
	  'attr { key: "w_scale" value { f: %f } }' % scale_value,
	  'attr { key: "num_classes" value { i: %d } }' % params['num_classes']
	]
	implements_signature = ' '.join(implements_signature)

	return implements_signature

def postprocess_tflite(params, cls_outputs, box_outputs, anchors, embeddings):
	"""Post processing for conversion to TFLite.
	Mathematically same as postprocess_global, except that the last portion of the
	TF graph constitutes a dummy `tf.function` that contains an annotation for
	conversion to TFLite's custom NMS op. Using this custom op allows features
	like post-training quantization & accelerator support.
	NOTE: This function does NOT return a valid output, and is only meant to
	generate a SavedModel for TFLite conversion via MLIR.
	For TFLite op details, see tensorflow/lite/kernels/detection_postprocess.cc
	Args:
	params: a dict of parameters.
	cls_outputs: a list of tensors for classes, each tensor denotes a level of
	  logits with shape [1, H, W, num_class * num_anchors].
	box_outputs: a list of tensors for boxes, each tensor ddenotes a level of
	  boxes with shape [1, H, W, 4 * num_anchors]. Each box format is [y_min,
	  x_min, y_max, x_man].
	Returns:
	A (dummy) tuple of (boxes, scores, classess, valid_len).
	"""

	decoded_anchors = anchors.get_anchors() # normalized anchors
	scores = tf.math.sigmoid(cls_outputs)

	box_indices = tf.argsort(scores[...,1], axis=-1, direction='DESCENDING')[..., :params["nms_configs"]["max_nms_inputs"]]
	scores = tf.reshape(tf.gather(scores, box_indices, axis=1), [scores.shape[0], -1, scores.shape[-1]])
	box_outputs = tf.reshape(tf.gather(box_outputs, box_indices, axis=1), [box_outputs.shape[0], -1, box_outputs.shape[-1]])
	embeddings = tf.reshape(tf.gather(embeddings, box_indices, axis=1), [embeddings.shape[0], -1, embeddings.shape[-1]])
	decoded_anchors = tf.gather(decoded_anchors, box_indices)[0] ## BE AWARE that because we drop the BATCH_DIM, we will specialize the exported model to BS=1 inference 


	# There is no TF equivalent for TFLite's custom post-processing op.
	# So we add an 'empty' composite function here, that is legalized to the
	# custom op with MLIR.
	# For details, see:
	# tensorflow/compiler/mlir/lite/utils/nms_utils.cc
	@tf.function(experimental_implements=tflite_nms_implements_signature(params))
	# pylint: disable=g-unused-argument,unused-argument
	def dummy_post_processing(box_encodings, class_predictions, anchor_boxes):
		boxes = tf.constant(0.0, dtype=tf.float32, name='boxes')
		scores = tf.constant(0.0, dtype=tf.float32, name='scores')
		classes = tf.constant(0.0, dtype=tf.float32, name='classes')
		num_detections = tf.constant(0.0, dtype=tf.float32, name='num_detections')
		return boxes, classes, scores, num_detections

	bboxes,classes, scores, num_detections = tf.keras.layers.Lambda(lambda x: dummy_post_processing(x[0], x[1], x[2]))([box_outputs, scores, decoded_anchors])#[::-1]

	return bboxes*[128,128,128,128], classes, scores, num_detections
	#return dummy_post_processing(box_outputs, scores, decoded_anchors)[::-1]


def custom_postprocess(
	inputs, 
	threshold=0.1, 
	score_threshold=0.1  , 
	min_spike_length=15, 
	max_spike_length = 64, 
	max_detection=10,
	pad_to_max=False,
	anchor_generator = AnchorsMobileNet):


	"""

	Returns:
		Boxes normalized in format (x1,y1, x2,y2) however they need to be upscaled 
	"""

	boxes = inputs[0]
	embeddings = inputs[1]
	classes = inputs[2] #inputs[1]

	boxes = boxes[..., :4]

	anchors = anchor_generator()


	params = {
				"nms_configs": {"iou_thresh" : threshold, "score_thresh":score_threshold, "max_nms_inputs":100}, 
				"tflite_max_detections": max_detection,
				"num_classes" : 2
			 }


	if classes.shape[-1] > 1:
		classes = classes[..., 1]
	else:
		classes = tf.squeeze(classes, axis=-1)


	classes = classes[0]
	boxes = boxes[0]
	embeddings = embeddings[0]

	boxes = anchors.bbox_transform_inv(boxes, approx=True)
	boxes = to_corners(boxes)

	selected_indices, selected_scores, valid_outputs =  tf.keras.layers.Lambda(lambda x: tf.raw_ops.NonMaxSuppressionV5(boxes=x[0],scores=x[1],max_output_size=x[2],iou_threshold=x[3],score_threshold=x[4], pad_to_max_output_size=pad_to_max,soft_nms_sigma = 0.0))([boxes, classes, max_detection, threshold, score_threshold])

	boxes = tf.gather(boxes, selected_indices)
	embeddings = tf.gather(embeddings, selected_indices)

	return boxes, embeddings, valid_outputs, selected_indices


def save_model_tflite(
	model, 
	save_path, 
	input_shape=(128,128,3), 
	iou_threshold=0.1, 
	score_threshold=0.1  
	):
	from .losses import EffDetLoss

	print("Saving as tflite model")

	newInput = tf.keras.layers.Input(shape=input_shape, batch_size=1)
	
	newOutputs = model(newInput)
	final_output = custom_postprocess(newOutputs, threshold=iou_threshold, score_threshold = score_threshold)

	newModel = tf.keras.Model(newInput,final_output)


	newModel.build(input_shape=(1,)+input_shape)


	converter = tf.lite.TFLiteConverter.from_keras_model(newModel) # path to the SavedModel directory
	converter.optimizations = [tf.lite.Optimize.DEFAULT]
	converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]

	tflite_model = converter.convert()

	if not save_path.endswith(".tflite"):
		save_path = save_path + ".tflite"

	# Save the model.
	with open(save_path, 'wb') as f:
	  f.write(tflite_model)

def save_model_quantized(
	model, 
	save_path, 
	representative_dataset, 
	dataset_loop=1, 
	input_shape=(128,128,3), 
	iou_threshold=0.1, 
	score_threshold=0.1,
	pseudo_batch = 1
	):
	from .losses import EffDetLoss

	print("Saving as quantized tflite model")


	newInput = [tf.keras.layers.Input(shape=input_shape, batch_size=1) for i in range(pseudo_batch)]
	
	newOutputs = [model(newInput[i]) for i in range(pseudo_batch)]
	final_output = [custom_postprocess(newOutputs[i], threshold=iou_threshold, score_threshold = score_threshold ) for i in range(pseudo_batch)] 

	newModel = tf.keras.Model(newInput,final_output)

	newModel.build(input_shape=[(1,)+input_shape for i in range(pseudo_batch)])


	def repr_dataset():
		rep_ds = iter(representative_dataset)
		for _ in range(dataset_loop):
			sample = next(rep_ds)[0]
			for i in range(sample.shape[0]):
				#sample = tf.concat(sample[i:i+1,:,32:64, :], sample[i:i+1,:,32:96,:], sample[i:i+1,:,64:96,:]], axis=-2)
				yield [sample[i:i+1] for i in range(pseudo_batch)]

	converter = tf.lite.TFLiteConverter.from_keras_model(newModel) # path to the SavedModel directory
	converter.optimizations = [tf.lite.Optimize.DEFAULT]
	converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
	converter.inference_input_type = tf.uint8

	converter.representative_dataset = repr_dataset

	tflite_model = converter.convert()

	if not save_path.endswith(".tflite"):
		save_path = save_path + ".tflite"

	# Save the model.
	with open(save_path, 'wb') as f:
	  f.write(tflite_model)	
