import tensorflow as tf
import numpy as np 
import time 

def tflite_export(
			 full_model_path, 
			 target_path
			 ):



	converter = tf.lite.TFLiteConverter.from_saved_model(full_model_path) # path to the SavedModel directory
	converter.optimizations = [tf.lite.Optimize.DEFAULT]
	converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]



	tflite_model = converter.convert()

	if not target_path.endswith(".tflite"):
		target_path = target_path + ".tflite"

	# Save the model.
	with open(target_path, 'wb') as f:
	  f.write(tflite_model)

def quantize_model(
			 full_model_path, 
			 target_path, 
			 representative_dataset, 
			 dataset_loop=100
			 ):

	def repr_dataset():
		for _ in range(dataset_loop):
			sample = next(iter(representative_dataset))[0]
			for i in range(sample.shape[0]):
				#sample = tf.concat(sample[i:i+1,:,32:64, :], sample[i:i+1,:,32:96,:], sample[i:i+1,:,64:96,:]], axis=-2)
				yield [sample[i:i+1]]

	converter = tf.lite.TFLiteConverter.from_saved_model(full_model_path) # path to the SavedModel directory
	converter.optimizations = [tf.lite.Optimize.DEFAULT]
	converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
	converter.representative_dataset = repr_dataset
	converter.inference_input_type = tf.uint8
	converter.inference_output_type = tf.uint8

	tflite_model = converter.convert()

	if not target_path.endswith(".tflite"):
		target_path = target_path + ".tflite"

	# Save the model.
	with open(target_path, 'wb') as f:
	  f.write(tflite_model)

class TFLiteModel:

	def __init__(self, tflite_file):

		self.quantization = False # initial values, later are changed if its the case 
		self.quantization = False

		self.interpreter = tf.lite.Interpreter(model_path=str(tflite_file))
		self.interpreter.allocate_tensors()

		input_details = self.interpreter.get_input_details()[0]

		box_output_details = self.interpreter.get_output_details()[0]
		cls_output_details = self.interpreter.get_output_details()[1]
		
		if input_details['dtype'] == np.uint8:
			self.quantization = True
			self.input_scale, self.input_zero_point = input_details["quantization"]

		if box_output_details['dtype'] == np.uint8:
			self.dequantization = True
			self.box_scale, self.box_zero_point = box_output_details["quantization"]
			self.cls_scale, self.cls_zero_point = cls_output_details["quantization"]   

		self.input_tensor = input_details["index"]
		self.box_output_tensor = box_output_details["index"]
		self.cls_output_tensor = cls_output_details["index"]

	def quantize(self, inp):
		
		if not self.quantization:
			return inp 

		inp = inp / self.input_scale + self.input_zero_point
		inp = tf.cast(inp, tf.uint8)
		return inp

	def dequantize_box(self, boxes):

		if not self.dequantization:
			return boxes

		boxes = tf.cast(boxes, tf.float32)
		boxes = (boxes - self.box_zero_point) * self.box_scale
		return boxes

	def dequantize_cls(self, classes):

		if not self.dequantization: 
			return classes

		classes = tf.cast(classes, tf.float32)
		classes = (classes - self.cls_zero_point) * self.cls_scale
		return classes


	def __call__(self, x):

		assert len(x.shape) == 4 # we should always have input as ( batch, height, width, channels)
		
		boxes = []
		classes = []

		for sample in x:

			b,c = self.__single_sample_infer(sample)

			if isinstance(boxes, list):
				boxes = b
				classes = c
			else:
				boxes = tf.concat([boxes, b], axis=0)
				classes = tf.concat([classes, c], axis=0)

		return boxes, classes

	def __single_sample_infer(self, x):

		x = self.quantize(x)
		self.interpreter.set_tensor(self.input_tensor, x)

		self.interpreter.invoke()

		boxes = self.interpreter.get_tensor(self.box_output_tensor)
		classes = self.interpreter.get_tensor(self.cls_output_tensor)

		boxes = self.dequantize_box(boxes)
		classes = self.dequantize_cls(classes)

		return boxes, classes
