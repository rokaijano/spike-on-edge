import tensorflow as tf
import numpy as np
import os
from contextlib import nullcontext
from random import shuffle
import matplotlib.pyplot as plt 
from matplotlib import patches

from object_detection.utils import dataset_util
from object_detection.core import standard_fields as fields

from model.anchors import *
from model.utils import to_xywh
from model.anchors import Anchors, AnchorsMobileNet

import tensorflow_addons as tfa
import tensorflow_probability as tfp


def scaling(x, sigma=0.1):
    # https://arxiv.org/pdf/1706.00527.pdf

    factor = tf.random.normal((x.shape[0], x.shape[1], x.shape[-1]), mean=1., stddev=sigma, dtype=tf.dtypes.float32)

    x = tf.cast(x, tf.float32)
    
    return tf.math.multiply(x, factor[:,:, np.newaxis,:])


def jitter(x, sigma=0.03):
    # https://arxiv.org/pdf/1706.00527.pdf
    return x + tf.random.normal(x.shape, mean=0., stddev=sigma)

class SpikeDataset():
    def __init__(self, 
                 file_path, 
                 batch_size=32, 
                 anchor_generator=AnchorsMobileNet, 
                 shuffle_buffer=128, 
                 target_embedding=False,
                 feature_len=32,
                 shuffle_files = True,
                 min_snr = 0,
                 use_cache = False, 
                 augmentation=False, 
                 with_batch=True
                 ):

        self._snap_timespan = 128
        self._target_channel_num = 128
        self._shuffle_buffer = shuffle_buffer
        self._batch_size = batch_size
        self._phase = 0
        self._one_hot = False
        self._feature_len = feature_len
        self._max_bb = 128
        self._anchor_num = self._target_channel_num * self._snap_timespan
        self._num_classes = 75
        self._target_pad = 341
        self._num_random_cutouts = 1
        self._augmentation = False
        self._target_embedding = target_embedding
        self.se = SamplesEncoder(aspect_ratios=[0.13], scales=[1./100], anchor_generator=anchor_generator)
        self._shuffle_files = shuffle_files
        self._min_snr = min_snr
        self._use_cache = use_cache
        self._augmentation = augmentation
        self._with_batch = with_batch

        # for compatibility
        self.num_anchors = self._anchor_num 

        self._waveforms = []
        self._waveforms_loaded = False


        file_list = self.prepare_files(file_path)

        if self._shuffle_files:
            shuffle(file_list)

        self._dataset = self.build_from_files(file_list)


    def __call__(self):
        return self._dataset
        
    # for compatibility
    def num_classes(self):
        return self._num_classes


    def create_flow_map(self, x):

        fr_multiplier = np.random.uniform(size=1) / 10.
        amplitude = np.random.uniform(high=2.0, size=1)

        f = np.random.normal(size=(self._batch_size,self._snap_timespan,self._target_channel_num,2)) * 3
        f[...,0] = 0

        for i in range(self._snap_timespan):
            k = np.sin(self._phase*fr_multiplier)*amplitude 
            f[:,i,:,1] = k

            self._phase = self._phase + 1

        return tfa.image.dense_image_warp(x, f)


    def prepare_files(self, file_paths):
        gt_files = []

        if self._target_embedding:
            self.load_waveforms(file_paths)

        for folder in file_paths:
            #data_dir = os.path.join(self.data_config["save_dir"], save_name)
            tf_files = os.listdir(folder)
            gt_files.extend([os.path.join(folder, x) for x in tf_files if x.endswith("gt.tfrecord")])

        return gt_files

    def augment_image(self, image):

        image = scaling(image)
        image = jitter(image)

        return image

    def load_waveforms(self, file_paths):
        """
        Fills the self._waveforms dictionary as such: 

            key : dataset name 
            value : array of shape [cluster_id, sizeof_waveform]
        
        """
        if self._waveforms_loaded:
            return


        waveform_dirs = [os.path.join(x, "waveform_embeddings") for x in file_paths]
        dataset_names = [x.split("\\")[-1] for x in file_paths]

        for i, wdir in enumerate(waveform_dirs):

            w_files = os.listdir(wdir)
            w_files = [x for x in w_files if x.endswith(".npy")]

            w_files = sorted(w_files, key=self.getint)

            self._waveforms.append([])

            self._waveforms[-1].append(np.zeros(self._feature_len))


            for w_id, wfile in enumerate(w_files):

                unit_id = int(os.path.splitext(wfile)[0])

                assert w_id == unit_id, f"w_id is {w_id} the other one, unit_id={unit_id}"

                w_full = os.path.join(wdir, wfile)

                self._waveforms[-1].append(np.load(w_full))


            self._waveforms[-1] = np.array(self._waveforms[-1])

        lookup_values = np.array(list(range(len(self._waveforms))), dtype=np.int64)

        self._waveform_lookup = tf.lookup.StaticVocabularyTable(
            initializer=tf.lookup.KeyValueTensorInitializer(
                keys=tf.constant(dataset_names),
                values=tf.constant(lookup_values),
            ),
            num_oov_buckets=1,
            name="waveform_embeddings"
        )

        self._waveforms = tf.constant(self._waveforms) 
        self._waveforms_loaded = True

    def parser_fn(self, proto):
        
        features = {
        'image/height': tf.io.FixedLenFeature([], tf.int64),
        'image/width': tf.io.FixedLenFeature([], tf.int64),
        'image/filename': tf.io.VarLenFeature(tf.string),
        'image/source_id': tf.io.VarLenFeature(tf.string),
        'image/encoded':  tf.io.FixedLenFeature([], tf.string),
        'image/format': tf.io.VarLenFeature(tf.string),
        'image/object/bbox/xmin': tf.io.VarLenFeature(tf.float32),
        'image/object/bbox/xmax': tf.io.VarLenFeature(tf.float32),
        'image/object/bbox/ymin': tf.io.VarLenFeature(tf.float32),
        'image/object/bbox/ymax': tf.io.VarLenFeature(tf.float32),
        'image/object/class/label': tf.io.VarLenFeature(tf.float32)
        }

        parsed_features = tf.io.parse_single_example(proto, features)
        
        im_h = tf.cast(parsed_features["image/height"], tf.int32)
        im_w = tf.cast(parsed_features["image/width"], tf.int32)

        dataset = parsed_features["image/source_id"] 

        
        image = tf.io.decode_raw(parsed_features['image/encoded'], tf.float32)
        
        image = tf.reshape(image, (im_h, im_w, 1))
        
        xmin = tf.expand_dims(tf.sparse.to_dense(parsed_features["image/object/bbox/xmin"]), axis=-1) + tf.cast((self._target_channel_num - im_w)//2, tf.float32)
        xmax = tf.expand_dims(tf.sparse.to_dense(parsed_features["image/object/bbox/xmax"]), axis=-1) + tf.cast((self._target_channel_num - im_w)//2, tf.float32)
        ymin = tf.expand_dims(tf.sparse.to_dense(parsed_features["image/object/bbox/ymin"]), axis=-1) 
        ymax = tf.expand_dims(tf.sparse.to_dense(parsed_features["image/object/bbox/ymax"]), axis=-1)
        class_labels = tf.sparse.to_dense(parsed_features["image/object/class/label"])


        bboxes = tf.concat([xmin,ymin, xmax, ymax], axis=-1)

        
        """

        Labels are as following: 
            -1 - is for fake labels for class 0 ( background )
            0  - are not used anchors (pruned in the following lines)
            >0 - are boxes with real activity in them 

        """
        valid_boxes = tf.logical_or(tf.greater(class_labels, 0), tf.equal(class_labels, -1))

        class_labels = tf.boolean_mask(class_labels, valid_boxes)
        bboxes = tf.boolean_mask(bboxes, valid_boxes)

        class_labels = tf.where(tf.equal(class_labels, -1.0), 0.0, class_labels)

        # Anchor generators` encode_sample func needs the stuff in center points xywh
        bboxes = to_xywh(bboxes)    
        
        image = (image - tf.math.reduce_mean(image)) / tf.math.reduce_std(image)

        image = tf.concat([image, image,image], axis=-1)

        class_labels = tf.reshape(class_labels, (-1,))

        """
        At this point class labels are as following: 

            
            0 - should be detected as a negative sample 
            >0 - should be detected as a positive sample 

        """

        ## here we define new width and height if its the case 

        im_h = 128#tf.math.maximum(im_h, im_w)
        im_w = 128#tf.math.maximum(im_h, im_w)

        ## padding the stuff 
        image = tf.image.resize_with_crop_or_pad(image, im_h, im_w)

        bboxes = tf.reshape(bboxes, (-1, 4))

        bboxes = bboxes / [[im_w,im_h,im_w,im_h]]

        
        box_target, class_target = self.se._encode_sample([1,im_h,im_w], bboxes, tf.cast(class_labels, tf.int32))

        box_target = tf.concat([box_target, class_target], axis=-1) # we concat in order to keep track of class target in the box branch as well

        class_target_embed = []

        if self._target_embedding:

            dataset_idx = self._waveform_lookup.lookup(dataset)
            dataset_idx = tf.sparse.to_dense(dataset_idx)[0]
            target_waveform = self._waveforms[dataset_idx]

            label_alias = tf.where(tf.less(class_target, 1.0), 0.0, class_target)
            target_waveform = tf.gather(target_waveform, tf.cast(tf.squeeze(label_alias, axis=1), tf.int32) ) # we have a 0-indexing waveform ds
            target_waveform = tf.cast(target_waveform, tf.float32)


            class_target_embed = tf.concat([class_target, target_waveform], axis=-1)

        return image, (box_target, class_target_embed, class_target)

    def interleave_fn(self, filename):
        dataset = tf.data.TFRecordDataset(filename)
        dataset = dataset.map(self.parser_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        return dataset

    def getint(self, name):
        return int(os.path.splitext(name)[0])

    def build_from_files(self, file_list):

        dataset = tf.data.Dataset.from_tensor_slices(file_list)
        dataset = dataset.interleave(self.interleave_fn, cycle_length=10, block_length=16, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        #dataset = dataset.repeat()

        if self._use_cache:
            dataset = dataset.cache()

        dataset = dataset.shuffle(buffer_size=self._shuffle_buffer, reshuffle_each_iteration=True) # for proper training 100000

        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

        if self._with_batch:
            dataset = dataset.batch(self._batch_size, drop_remainder=True)

            """
            AUGMENTATIONS 
            """
            if self._augmentation:
                dataset = dataset.map(lambda x,y: (self.augment_image(x), y))


        return dataset



