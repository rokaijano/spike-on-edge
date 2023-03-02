import tensorflow as tf
import numpy as np
import os
from contextlib import nullcontext
from random import shuffle
import matplotlib.pyplot as plt 
from matplotlib import patches

from object_detection.utils import dataset_util
from object_detection.core import standard_fields as fields

import tensorflow_addons as tfa
import tensorflow_probability as tfp

@tf.autograph.experimental.do_not_convert
def to_xywh(bbox):
    """Convert [x_min, y_min, x_max, y_max] to [x, y, width, height]."""
    return tf.concat(
        [(bbox[..., :2] + bbox[..., 2:]) / 2.0, (bbox[..., 2:] - bbox[..., :2])], axis=-1
    )

def permutation(x, max_segments=5, seg_mode="equal"):

    orig_steps = tf.experimental.numpy.arange(x.shape[1])
    
    num_segs = tf.experimental.numpy.random.randint(1, max_segments, size=(x.shape[0]))
    
    ret = tf.zeros_like(x)
    for i, pat in enumerate(x):
        if num_segs[i] > 1:
            if seg_mode == "random":
                split_points = np.random.choice(x.shape[1]-2, num_segs[i]-1, replace=False)
                split_points.sort()
                splits = tf.split(orig_steps, split_points)
            else:
                splits = np.array_split(orig_steps, num_segs[i])
            warp = np.concatenate(np.random.permutation(splits)).ravel()
            ret[i] = pat[warp]
        else:
            ret[i] = pat
    return ret


def scaling(x, sigma=0.1):
    # https://arxiv.org/pdf/1706.00527.pdf

    factor = tf.random.normal((x.shape[0],x.shape[2]), mean=1., stddev=sigma, dtype=tf.dtypes.float32)

    x = tf.cast(x, tf.float32)
    
    return tf.math.multiply(x, factor[:,np.newaxis,:])


def jitter(x, sigma=0.03):
    # https://arxiv.org/pdf/1706.00527.pdf
    return x + tf.random.normal((x.shape[0],105, x.shape[2]), mean=0., stddev=sigma)

def magnitude_warp(x, sigma=0.2, knot=4):
    from scipy.interpolate import CubicSpline
    orig_steps = tf.experimental.numpy.arange(x.shape[1])
    
    random_warps = tf.random.normal((x.shape[0], knot+2, x.shape[2]), mean=1.0, stddev=sigma)
    warp_steps = (tf.ones((x.shape[2],1))*(np.linspace(0, x.shape[1]-1., num=knot+2))).T
    ret = np.zeros_like(x)
    for i, pat in enumerate(x):
        warper = np.array([CubicSpline(warp_steps[:,dim], random_warps[i,:,dim])(orig_steps) for dim in range(x.shape[2])]).T
        ret[i] = pat * warper

    return ret

class WaveFormDataset():
    def __init__(self,
                 file_path, 
                 with_batch=True,
                 supervised=False,
                 batch_size=32, 
                 shuffle_buffer=100000, 
                 feature_len=128,
                 shuffle_files = True,
                 min_snr = 0,
                 use_cache = True,
                 augmentation = False
                 ):

        self._with_batch = with_batch
        self._file_path = file_path
        self._supervised = supervised
        self._snap_timespan = 128
        self._target_channel_num = 128
        self._shuffle_buffer = max(shuffle_buffer, batch_size*4) 
        self._batch_size = batch_size
        self._phase = 0
        self._feature_len = feature_len
        self._augmentation = augmentation
        self._waveforms_loaded = False 
        self._one_hot_classes = 75
        self._shuffle_files = shuffle_files
        self._min_snr = min_snr
        self._use_cache = use_cache

        # for compatibility
        self._waveforms = []


        file_list = self.prepare_files(file_path)
        
        if self._supervised:
            self._one_hot_classes = len(file_list)

        if self._shuffle_files:
            shuffle(file_list)

        self._dataset = self.build_from_files(file_list)

    def __call__(self):
        return self._dataset

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

        self.load_waveforms(file_paths)

        for folder in file_paths:
            #data_dir = os.path.join(self.data_config["save_dir"], save_name)
            tf_files = os.listdir(folder)
            gt_files.extend([os.path.join(folder, x) for x in tf_files if x.endswith("gt.tfrecord")])

        return gt_files

    def load_waveforms(self, file_paths):
        """
        Fills the self._waveforms dictionary as such: 

            key : dataset name 
            value : array of shape [cluster_id, sizeof_waveform]
        
        """
        if self._waveforms_loaded:
            return

        longest_list = 0

        waveform_dirs = [os.path.join(x, "waveforms/") for x in file_paths]
        dataset_names = [x.split("\\")[-1] for x in file_paths]

        for i, wdir in enumerate(waveform_dirs):

            w_files = os.listdir(wdir)
            w_files = [x for x in w_files if x.endswith(".npy")]


            w_files = sorted(w_files, key=self.getint)

            self._waveforms.append([])
            for w_id, wfile in enumerate(w_files):

                unit_id = int(os.path.splitext(wfile)[0])

                assert w_id == unit_id, f"w_id is {w_id} the other one, unit_id={unit_id}"

                w_full = os.path.join(wdir, wfile)

                self._waveforms[-1].append(np.load(w_full))


            if longest_list < len(self._waveforms[-1]):
                longest_list = len(self._waveforms[-1])

        lookup_values = np.array(list(range(len(self._waveforms))), dtype=np.int64)

        self._waveform_lookup = tf.lookup.StaticVocabularyTable(
            initializer=tf.lookup.KeyValueTensorInitializer(
                keys=tf.constant(dataset_names),
                values=tf.constant(lookup_values),
            ),
            num_oov_buckets=1,
            name="average_waveforms"
        )
        
        for i, wave_dataset in enumerate(self._waveforms):

            diff = longest_list - len(wave_dataset)
            for k in range(diff):
                self._waveforms[i].append(np.zeros(len(wave_dataset[0])))
         

        self._waveforms = tf.constant(self._waveforms) # shape is ( dataset , (), ())
        self._waveforms_loaded = True


    def getint(self, name):
        return int(os.path.splitext(name)[0])

    def augment_image(self, image):

        #image = permutation(image)
        image = scaling(image)
        #image = magnitude_warp(image)
        image = jitter(image)

        return image

    def get_real_median(self, v):

        v = tf.reshape(v, [-1])
        l = v.get_shape()[0]
        mid = l//2 + 1
        val = tf.nn.top_k(v, mid).values
        if l % 2 == 1:
            return val[-1]
        else:
            return 0.5 * (val[-1] + val[-2])

    def compute_snr(self, template):

        ym_hat = tf.math.reduce_mean(template)

        mad_m = self.get_real_median(tf.abs(template))

        std_m = mad_m / 0.6745

        snr = tf.math.reduce_max(tf.math.abs(template)) / std_m

        return snr

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

        dataset = parsed_features["image/source_id"] ## #####
        filename = tf.sparse.to_dense(parsed_features["image/filename"]) ## #####
        
        image = tf.io.decode_raw(parsed_features['image/encoded'], tf.float32)
        
        image = tf.reshape(image, (im_h, im_w, 1))
        
        xmin = tf.expand_dims(tf.sparse.to_dense(parsed_features["image/object/bbox/xmin"]), axis=-1)
        xmax = tf.expand_dims(tf.sparse.to_dense(parsed_features["image/object/bbox/xmax"]), axis=-1)
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
        valid_boxes = tf.greater(class_labels, 0)

        class_labels = tf.boolean_mask(class_labels, valid_boxes)
        bboxes = tf.boolean_mask(bboxes, valid_boxes)

        filename = tf.strings.split(filename, sep="\\")[-1]
        filename = tf.strings.split(filename, sep="/")[-1]

        filename = tf.strings.regex_replace(filename, "_gt.tfrecord", "")

        class_id = tf.strings.to_number(filename, tf.dtypes.float32) + 1

        valid_boxes = tf.equal(class_labels, class_id)

        center_box = tf.boolean_mask(bboxes, valid_boxes)
        class_labels = tf.boolean_mask(class_labels, valid_boxes)
        
        class_labels = class_labels[-1]
        center_box = center_box[-1]

        dataset_idx = self._waveform_lookup.lookup(dataset)
        dataset_idx = tf.sparse.to_dense(dataset_idx)[0]

        target_waveform = self._waveforms[dataset_idx]
        

        class_labels = tf.cast(class_labels, tf.int32)-1
        target_waveform = tf.gather(target_waveform, class_labels)

        # we add to them the necessary amount to get the center of activation to align to target
        ch = int((center_box[0] + center_box[2]) / 2) # channel dimension
        t = int(center_box[1]) + 10 # timestep dimension

        end_time = tf.math.minimum(tf.shape(image)[0], t+60)

        sample = image[end_time-tf.shape(target_waveform)[0]:end_time, ch]

        sample = tf.squeeze(sample)

        w_min = tf.math.reduce_min(target_waveform)
        w_max = tf.math.reduce_max(target_waveform)

        snr = self.compute_snr(target_waveform)

        target_waveform = (target_waveform - w_min) / (w_max - w_min)


        s_min = tf.math.reduce_min(sample)
        s_max = tf.math.reduce_max(sample)

        sample = (sample - s_min) / (s_max - s_min)

        sample = tf.reshape(sample, (-1,1))
        target_waveform = tf.reshape(target_waveform, (-1,1))

        if self._supervised:
            class_labels = tf.one_hot(class_labels, self._one_hot_classes)
            return (sample, ch), class_labels, snr

        return sample, target_waveform, snr

    def interleave_fn(self, filename):
        dataset = tf.data.TFRecordDataset(filename)
        dataset = dataset.map(self.parser_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        return dataset



    def build_from_files(self, file_list):
        dataset = tf.data.Dataset.from_tensor_slices(file_list)
        dataset = dataset.interleave(self.interleave_fn, cycle_length=16, block_length=1, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        #dataset = dataset.repeat()

        if self._min_snr > 0:
            dataset = dataset.filter(lambda x,y,snr : snr  >= self._min_snr)

        #dataset = dataset.map(lambda x,y,snr: (x,y))

        if self._use_cache:
            dataset = dataset.cache()

        dataset = dataset.shuffle(buffer_size=self._shuffle_buffer, reshuffle_each_iteration=True) # for proper training 100000
        
        

        if self._with_batch:
            dataset = dataset.batch(self._batch_size, drop_remainder=True)



        if self._augmentation:
            dataset = dataset.map(lambda x,y,snr: (self.augment_image(x), self.augment_image(y), snr))

        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

        return dataset