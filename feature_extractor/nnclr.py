

import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import backend as K


from encoder import Encoder

import tensorflow.python.autograph as autograph


AUTOTUNE = tf.data.AUTOTUNE
shuffle_buffer = 5000
# The below two values are taken from https://www.tensorflow.org/datasets/catalog/stl10
labelled_train_images = 5000
unlabelled_images = 100000


contrastive_augmenter = {
	"brightness": 0.15,
	"name": "contrastive_augmenter",
	"scale": (0.2, 1.0),
}
classification_augmenter = {
	"brightness": 0.1,
	"name": "classification_augmenter",
	"scale": (0.5, 1.0),
}
input_shape = (105,1)
width = 32


"""
### Random Resized Crops
"""


class RandomResizedCrop(layers.Layer):
	def __init__(self, scale, ratio):
		super(RandomResizedCrop, self).__init__()
		self.scale = scale
		self.log_ratio = (tf.math.log(ratio[0]), tf.math.log(ratio[1]))

	def call(self, images):
		batch_size = tf.shape(images)[0]
		height = tf.shape(images)[1]
		width = tf.shape(images)[2]

		random_scales = tf.random.uniform((batch_size,), self.scale[0], self.scale[1])
		random_ratios = tf.exp(
			tf.random.uniform((batch_size,), self.log_ratio[0], self.log_ratio[1])
		)

		new_heights = tf.clip_by_value(tf.sqrt(random_scales / random_ratios), 0, 1)
		new_widths = tf.clip_by_value(tf.sqrt(random_scales * random_ratios), 0, 1)
		height_offsets = tf.random.uniform((batch_size,), 0, 1 - new_heights)
		width_offsets = tf.random.uniform((batch_size,), 0, 1 - new_widths)

		bounding_boxes = tf.stack(
			[
				height_offsets,
				width_offsets,
				height_offsets + new_heights,
				width_offsets + new_widths,
			],
			axis=1,
		)
		images = tf.image.crop_and_resize(
			images, bounding_boxes, tf.range(batch_size), (height, width)
		)
		return images


"""
### Random Brightness
"""


class RandomBrightness(layers.Layer):
	def __init__(self, brightness, **kwargs):
		super(RandomBrightness, self).__init__(**kwargs)
		self.brightness = brightness

	def blend(self, images_1, images_2, ratios, const):
		
		newim = tf.clip_by_value(const * images_1 + (1.0 - const) * images_2, 0, 1)
		
		return tf.clip_by_value(ratios * newim + (1.0 - ratios) * images_2, 0, 1)

	def random_brightness(self, images):
		# random interpolation/extrapolation between the image and darkness
		return self.blend(
			images,
			0,
			tf.random.uniform(
				(tf.shape(images)[0], tf.shape(images)[1], 1), 1 - self.brightness, 1 + self.brightness
			),
			tf.random.uniform(
				(tf.shape(images)[0], 1, 1), 1 - self.brightness, 1 + self.brightness
			)
		)

	def call(self, images):
		images = self.random_brightness(images)
		return images

	def get_config(self):
		config = super().get_config()
		config.update({
			"brightness": self.brightness
			})
		return config


"""
### Prepare augmentation module
"""


def augmenter(brightness, name, scale):
	return keras.Sequential(
		[
			layers.Input(shape=input_shape),
			RandomBrightness(brightness=brightness),
		],
		name=name,
	)

@autograph.convert()
def filter_negatives(x,y):

	negs_x = x[y==0]
	negs_y = y[y==0]

	x = x[y!=0]
	y = y[y!=0]

	x = tf.concat([x, negs_x[:len(x)//5]], axis=0)
	y = tf.concat([y, negs_y[:len(y)//5]], axis=0)
	return x,y

#"""
def encoder():

	inp = tf.keras.layers.Input(shape=input_shape)
	x = inp

	x = ResidualBlock(64, _strides=3, name="enc_64_0")(x)

	x = ResidualBlock(128, _strides=3, name="enc_128_0")(x)

	x = ResidualBlock(256, _strides=3, name="enc_256_0")(x)

	x = tf.keras.layers.Conv1D(
		filters=256,
		kernel_size=1,
		strides=1,
		padding="same",
		kernel_initializer=tf.keras.initializers.HeNormal()
		)(x)

	x = layers.BatchNormalization()(x)
	x = layers.LeakyReLU()(x)

	x = tf.keras.layers.Conv1D(
		filters=128,
		kernel_size=1,
		strides=1,
		padding="same",
		kernel_initializer=tf.keras.initializers.HeNormal()
		)(x)

	x = layers.BatchNormalization()(x)
	x = layers.LeakyReLU()(x)

	x = tf.keras.layers.Conv1D(
		filters=64,
		kernel_size=1,
		strides=1,
		padding="same",
		kernel_initializer=tf.keras.initializers.HeNormal()
		)(x)

	x = layers.BatchNormalization()(x)
	x = layers.LeakyReLU()(x)

	x = tf.keras.layers.Conv1D(
	  filters=width,
	  kernel_size=1,
	  strides=1,
	  padding="same",
	  name="enc_quant_t", 
	  kernel_initializer=tf.keras.initializers.HeNormal()
	  )(x)

	x = layers.BatchNormalization()(x)
	x = layers.LeakyReLU()(x)
	x = layers.Flatten()(x)
	x = layers.Dense(width, kernel_initializer=tf.keras.initializers.HeNormal())(x)
	x = layers.BatchNormalization()(x)
	x = layers.LeakyReLU()(x)
	x = layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=-1))(x)
	

	model = tf.keras.Model(inp, x, name="encoder")
	print(model.summary())
	return model
#"""

def distinct(input:tf.Tensor) -> tf.Tensor:
	"""Returns only the distinct sub-Tensors along the 0th dimension of the provided Tensor"""
	is_equal = tf.equal(input[:,tf.newaxis], input[tf.newaxis,:])
	while len(is_equal.shape) > 2:
		is_equal = tf.math.reduce_all(is_equal, axis=2)
	all_true = tf.constant(True, shape=is_equal.shape)
	true_upper_tri = tf.linalg.band_part(all_true, 0, -1)
	false_upper_tri = tf.math.logical_not(true_upper_tri)
	is_equal_one_way = tf.math.logical_and(is_equal, false_upper_tri)
	is_duplicate = tf.reduce_any(is_equal_one_way, axis=1)
	is_distinct = tf.math.logical_not(is_duplicate)
	distinct_elements = tf.boolean_mask(input, is_distinct, 0)
	return distinct_elements
"""
## The NNCLR model for contrastive pre-training

We train an encoder on unlabeled images with a contrastive loss. A nonlinear projection
head is attached to the top of the encoder, as it improves the quality of representations
of the encoder.
"""


class NNCLR(keras.Model):
	def __init__(
		self, temperature=0.1, queue_size=10000,
	):
		super(NNCLR, self).__init__()
		self.probe_accuracy = keras.metrics.CategoricalAccuracy()
		self.correlation_accuracy = keras.metrics.SparseCategoricalAccuracy()
		self.contrastive_accuracy = keras.metrics.SparseCategoricalAccuracy()
		self.probe_loss = keras.losses.CategoricalCrossentropy(from_logits=True)
		self.queue_size = queue_size

		self.contrastive_augmenter = augmenter(**contrastive_augmenter)
		self.classification_augmenter = augmenter(**classification_augmenter)
		self.encoder = encoder()
		self.projection_head = keras.Sequential(
			[
				layers.Input(shape=(width,)),
				layers.Dense(width, kernel_initializer=tf.keras.initializers.HeNormal()),
				layers.BatchNormalization(),
				layers.Activation("relu"),
				layers.Dense(width)
			],
			name="projection_head",
		)
		self.linear_probe = keras.Sequential(
			[
			layers.Input(shape=(width,)),
			layers.Dense(width, kernel_initializer=tf.keras.initializers.HeNormal()),
			layers.BatchNormalization(),
			layers.Activation("relu"),
			layers.Dense(250)
			], name="linear_probe"
		)

		self.temperature = temperature

		feature_dimensions = width#self.encoder.output_shape[1]
		self.feature_queue = tf.Variable(
			tf.math.l2_normalize(
				tf.random.normal(shape=(queue_size, feature_dimensions)), axis=1
			),
			trainable=False,
		)

	def compile(self, contrastive_optimizer, probe_optimizer, **kwargs):
		super(NNCLR, self).compile(**kwargs)
		self.contrastive_optimizer = contrastive_optimizer
		self.probe_optimizer = probe_optimizer

	def nearest_neighbour(self, projections):
		support_similarities = tf.matmul(
			projections, self.feature_queue, transpose_b=True
		)
		nn_projections = tf.gather(
			self.feature_queue, tf.argmax(support_similarities, axis=1), axis=0
		)
		return projections + tf.stop_gradient(nn_projections - projections)

	def update_contrastive_accuracy(self, features_1, features_2):
		features_1 = tf.math.l2_normalize(features_1, axis=1)
		features_2 = tf.math.l2_normalize(features_2, axis=1)
		similarities = tf.matmul(features_1, features_2, transpose_b=True)

		batch_size = tf.shape(features_1)[0]
		contrastive_labels = tf.range(batch_size)
		self.contrastive_accuracy.update_state(
			tf.concat([contrastive_labels, contrastive_labels], axis=0),
			tf.concat([similarities, tf.transpose(similarities)], axis=0),
		)

	def update_correlation_accuracy(self, features_1, features_2):
		features_1 = (
			features_1 - tf.reduce_mean(features_1, axis=0)
		) / tf.math.reduce_std(features_1, axis=0)
		features_2 = (
			features_2 - tf.reduce_mean(features_2, axis=0)
		) / tf.math.reduce_std(features_2, axis=0)

		batch_size = tf.shape(features_1, out_type=tf.int32)[0]
		batch_size = tf.cast(batch_size, tf.float32)

		cross_correlation = (
			tf.matmul(features_1, features_2, transpose_a=True) / batch_size
		)

		feature_dim = tf.shape(features_1)[1]
		correlation_labels = tf.range(feature_dim)
		self.correlation_accuracy.update_state(
			tf.concat([correlation_labels, correlation_labels], axis=0),
			tf.concat([cross_correlation, tf.transpose(cross_correlation)], axis=0),
		)

	def contrastive_loss(self, projections_1, projections_2):


		batch_size = tf.shape(projections_1)[0]
		contrastive_labels = tf.range(batch_size)

		#""" EREDETI

		projections_1 = tf.math.l2_normalize(projections_1, axis=1)
		projections_2 = tf.math.l2_normalize(projections_2, axis=1)

		similarities_1_2_1 = (
			tf.matmul(
				self.nearest_neighbour(projections_1), projections_2, transpose_b=True
			)
			/ self.temperature
		)
		similarities_1_2_2 = (
			tf.matmul(
				projections_2, self.nearest_neighbour(projections_1), transpose_b=True
			)
			/ self.temperature
		)

		similarities_2_1_1 = (
			tf.matmul(
				self.nearest_neighbour(projections_2), projections_1, transpose_b=True
			)
			/ self.temperature
		)
		similarities_2_1_2 = (
			tf.matmul(
				projections_1, self.nearest_neighbour(projections_2), transpose_b=True
			)
			/ self.temperature
		)
		
		loss = keras.losses.sparse_categorical_crossentropy(
			tf.concat(
				[
					contrastive_labels,
					contrastive_labels,
					contrastive_labels,
					contrastive_labels,
				],
				axis=0,
			),
			tf.concat(
				[
					similarities_1_2_1,
					similarities_1_2_2,
					similarities_2_1_1,
					similarities_2_1_2,
				],
				axis=0,
			),
			from_logits=True,
		)

		self.feature_queue.assign(
			tf.concat([projections_2, self.feature_queue[:-batch_size]], axis=0)[:self.queue_size]
		)

		return loss
		#"""
		loss = keras.losses.sparse_categorical_crossentropy(
			contrastive_labels,
			tf.matmul(projections_1, projections_2, transpose_b=True) / self.temperature,
			from_logits=True,
		)
		return loss

	def train_step(self, data):

		image1 = data[0]
		image2 = data[1]


		#images = tf.concat((unlabeled_images, labeled_images), axis=0)
		augmented_images_1 = self.contrastive_augmenter(image1)
		augmented_images_2 = self.classification_augmenter(image2)##self.contrastive_augmenter(image2)

		with tf.GradientTape() as tape:
			features_1 = tf.reshape(self.encoder(augmented_images_1),[-1, width])
			features_2 = tf.reshape(self.encoder(augmented_images_2), [-1, width])

			projections_1 = self.projection_head(features_1)
			projections_2 = self.projection_head(features_2)
			contrastive_loss = self.contrastive_loss(projections_1, projections_2)


		gradients = tape.gradient(
			contrastive_loss,
			self.encoder.trainable_weights + self.projection_head.trainable_weights,
		)
		self.contrastive_optimizer.apply_gradients(
			zip(
				gradients,
				self.encoder.trainable_weights + self.projection_head.trainable_weights,
			)
		)
		self.update_contrastive_accuracy(features_1, features_2)
		self.update_correlation_accuracy(features_1, features_2)

		return {
			"loss": contrastive_loss,
			"c_acc": self.contrastive_accuracy.result(),
			"r_acc": self.correlation_accuracy.result()
		}
	

class ResidualBlock(tf.keras.layers.Layer):

	def __init__(self, nb_channels, _kernel=3, _strides=1, _project_shortcut=False, name="", **kwargs):

		super(ResidualBlock, self).__init__(**kwargs)

		self.project_shortcut = _project_shortcut
		self.strides = _strides
		self.nb_channels = nb_channels
		self.kernel = _kernel

		# down-sampling is performed with a stride of 2
		self.conv2d_1 = layers.Conv1D(nb_channels, kernel_size=self.kernel, strides=_strides, padding='same', name="c1_"+name, kernel_initializer=tf.keras.initializers.HeNormal())

		self.lrelu_1 = layers.LeakyReLU(name="lr1_"+name)
		self.conv2d_2 = layers.Conv1D(nb_channels, kernel_size=self.kernel, strides=1, padding='same', name="c2_"+name, kernel_initializer=tf.keras.initializers.HeNormal())

		self.batchnorms = [layers.BatchNormalization() for i in range(3)]

		# identity shortcuts used directly when the input and output are of the same dimensions
		if _project_shortcut or _strides != 1:
			# when the dimensions increase projection shortcut is used to match dimensions (done by 1×1 convolutions)
			# when the shortcuts go across feature maps of two sizes, they are performed with a stride of 2
			self.conv2d_1_shortcut = layers.Conv1D(nb_channels, kernel_size=1, strides=_strides, padding='same', name="s1_"+name, kernel_initializer=tf.keras.initializers.HeNormal())
			self.batchnorm_shortcut = layers.BatchNormalization(renorm=True, name="bs1_"+name)

		#y = layers.add([shortcut, y])
		self.lrelu_2 = layers.LeakyReLU(name="lr2_"+name)

	def call(self, x):
		x_orig = x 
		x = self.conv2d_1(x)
		x = self.batchnorms[0](x)
		x = self.lrelu_1(x)
		x = self.conv2d_2(x)
		x = self.batchnorms[1](x)

		if self.project_shortcut or self.strides != 1:
			x_orig = self.conv2d_1_shortcut(x_orig)
			x_orig = self.batchnorm_shortcut(x_orig)

		x = layers.add([x_orig, x])
		x = self.batchnorms[2](x)
		x = self.lrelu_2(x)

		return x 

	def get_config(self):
		return {
		"nb_channels": self.nb_channels, 
		"_strides":self.strides, 
		"_project_shortcut":self.project_shortcut,
		"_kernel":self.kernel
		}

	@classmethod
	def from_config(cls, config):
		return cls(**config)

