import os
import sys

import glob
import numpy as np
import xmltodict

import tensorflow as tf
from scipy.signal import butter, lfilter
from sklearn.model_selection import train_test_split

import gc
import multiprocessing
from multiprocessing import Pool
from multiprocessing import RawArray
from object_detection.utils import dataset_util


#import matplotlib.pyplot as plt 

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def butter_bandpass(lowcut, highcut, fs, order=5, btype="band"):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype=btype)
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5, axis=0, btype="band"):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order, btype=btype)
    y = lfilter(b, a, data, axis=axis)
    return y

def config_post(path, key, value):

    if key.endswith("value"):
        try:
            return key, int(value)
        except (ValueError, TypeError):
            return key, value

    elif key.endswith("bool"):
        try:
            return key, bool(value)
        except (ValueError, TypeError):
            return key, value

    return key, value

def geom_to_index(geom):
    x_unique = np.unique(geom[:, 0])
    y_unique = np.unique(geom[:, 1])
    
    for i,x in enumerate(x_unique):
        geom[:,0] = np.where(geom[:,0]==x, i, geom[:,0]) 
        
    for i,y in enumerate(y_unique):
        geom[:,1] = np.where(geom[:,1]==y, i, geom[:,1]) 

    return geom

class Preprocessor(object):
    
    def __init__(self, config_file ="", config={}, preloaded_channels=[], overwrite=False):
    
        if config_file != "" and path.exists(config_file):
            with open(config_file) as fd:
                try:
                    self.config = xmltodict.parse(fd.read(), dict_constructor=dict, postprocessor=config_post)
                    self.config = self.config["root"]
                    self.config.update(config)

                except Exception:
                    self.config = config
        else:
            self.config = config

        self.config.setdefault("save_dir", "data/")
        self.config.setdefault("save_name","") #
        self.config.setdefault("default_config_name", "config.xml")
        self.config.setdefault("default_waveform_dir", "waveforms/")
        self.config.setdefault("data_source", "")
        self.config.setdefault("channel_num", 128)
        self.config.setdefault("freq", 20000)
        self.config.setdefault("spike_timespan", 32)
        self.config.setdefault("snap_timespan", 128)
        self.config.setdefault("labeling_length", 15)
        self.config.setdefault("channel_bounding_box", 2)
        self.config.setdefault("overlap_limit", 5) # to remove or include spike from end/beginning of segment 
        self.config.setdefault("waveform_before_ms", 1.5)
        self.config.setdefault("waveform_after_ms", 2)
        self.config.setdefault("curate_channel_identification", True)

        self.preloaded_channels = preloaded_channels
        #self.config.setdefault("target_dim", 128) # target_dim = S = 128x128

        ## Preprocessing parameters

        self._nbefore_waveform = 0 # will be populated in preprocess_with_label fun
        self._nafter_waveform = 0 # will be populated in preprocess_with_label fun

        self.overwrite_tfrecords = overwrite
        self.var_dict = {}  


    def init_worker(self, X, X_shape, Y, Y_shape):
        # Using a dictionary is not strictly necessary. You can also
        # use global variables.
        self.var_dict['X'] = X
        self.var_dict['X_shape'] = X_shape
        self.var_dict['Y'] = Y
        self.var_dict['Y_shape'] = Y_shape

    def load_data_and_labels(self, filename):
        # should return a pair of numpy arrays of dimensions ( [timestep, channels], [neuron, activations] )
        raise NotImplementedError

    def run(self):
        ftrain = ""

        x,y = self.load_data_and_labels(ftrain)

        self.preprocess_with_label(x,y)

        with open(os.path.join(self.config["save_dir"], self.config["save_name"], self.config["default_config_name"]), "w") as fd:
            root = {"root":self.config}
            fd.write(xmltodict.unparse(root, pretty = True))

    def get_label_list(self, neuron_idx, snr =-1, timestep=-1):
        """
        Input: 
            self
            neuron_idx - index of the neuron that is needed for the label
            snr - unused atm
            A mask of the appearance of the neuron_idx from timestep

        """

        # this is the end of the timespan where a spike might start, but the whole waveform cannot be seen, so we do not label them
        death_zone = self.config["snap_timespan"] - self.config["overlap_limit"]

        x1 = -1
        y1 = -1
        x2 = -1 
        y2 = -1
        n_cls = -1        


        assert timestep >=-self.config["overlap_limit"] and neuron_idx >=0 and timestep<self.config["snap_timespan"] and neuron_idx < len(self.neuron_channels), "Incorrect label values"

        if  neuron_idx < 0 or timestep >= death_zone:
            return x1,y1,x2,y2, n_cls

        x1 = timestep
        y1 = self.neuron_channels[neuron_idx] - 1

        x2 = max(min(timestep+self.config["labeling_length"], self.config["snap_timespan"]-1),0)
        y2 = self.neuron_channels[neuron_idx] + 1

        n_cls = neuron_idx

        return x1,y1,x2,y2, n_cls

    def get_file_path(self, n_id):
        return os.path.join(self.config["save_dir"], self.config["save_name"], str(n_id)+"_gt.tfrecord")

    def setup_neuron_channels(self, neurons=[]):

        if len(neurons) == 0:
            neurons = self.neurons

        self.neuron_channels = np.ones((len(neurons),), dtype=np.int32) * -1 


        ##############
        #############
        ############
        #################

        for i, val in enumerate(self.preloaded_channels):
            self.neuron_channels[i] = val
        #####################
        #####################
        ####################
        ############55
        #############
        #########################

        print("- Computing the closest channel for each neuron ... ")

        #for n_id in range(len(neurons)):
        #    self.compute_neuron_channels(n_id)

    def compute_channel_for_neuron(self, n_id, max_element = 150, sample_size = 8000):

        print("---- Computing for neuron with id: "+str(n_id), end="\r")
        max_channels = []

        if not self.overwrite_tfrecords and os.path.isfile(self.get_file_path(n_id)):
            print(f"---- Skipping neuron with id of {n_id}, because already exists....")
            return  

        if self.neuron_channels[n_id] != -1: 
            print(f"---- Skipping neuron with id of {n_id}, because ch id was preloaded")
            return

        snap_list = None
        counter = 0

        divider = len(self.neurons[n_id]) if max_element is None else max_element 

        for i, t_spike in enumerate(self.neurons[n_id]): 
            if t_spike == 0: # because of the padding
                continue 

            if max_element != None and counter > max_element: # for lowering the processing time 
                break

            lower_bound = max((t_spike-sample_size//2), 0)

            center = t_spike - lower_bound

            upper_bound = min(lower_bound + sample_size, len(self.raw_data)-1) # attention!! in special cases the target spike won't be at the center

            sample = self.raw_data[lower_bound:upper_bound]

            sample = np.array(butter_bandpass_filter(sample, 300, 3000, self.config["freq"], 5, axis=0))

            snap = sample[center-50:center+50]

            ## for testing 
            if counter == 0:
                snap_list = snap / divider

            else:
                try:
                    snap_list = snap_list + snap / divider
                except ValueError:
                    pass

            counter = counter + 1 
            ## for testing end 

            #c = np.argmax(np.abs(snap[50]-snap[55]) * np.var(snap, axis=0))
            #max_channels.append(c)
            
        #y = np.bincount(max_channels)

        #snap_list = np.array(snap_list)

        #snap = np.mean(snap_list, axis=0)
        snap = snap_list ## this is already the mean 




        if self.config["dataset_name"] == "STATIC_8X_A_2A":
            part = snap[40:65]
        else:
            part = snap[45:55]    




        c = np.argmax(np.max(part, axis=0)-np.min(part, axis=0))
        
        
        #"""
        if self.config["curate_channel_identification"]:
            print(c)
            import matplotlib.pyplot as plt 

            input_ok = False

            while not input_ok:
                plt.plot(snap)
                plt.show()
                snap = np.expand_dims(snap, -1)
                plt.imshow(snap)
                plt.show()
                #"""

                try:
                    c = int(input("Channel number: "))

                    if c < 0:
                        continue

                except Exception:
                    continue

                input_ok = True

        self.neuron_channels[n_id] = c#np.argmax(y)

        gc.collect()
        print("---- Successfully identified all the channels", end="\r\n\n")
        


    def calc_SNR(self, spike): 
        amplitude = 0.

        # we assume the main channel is centered !! linearly !! 

        channel_idx = (spike.shape[1]-1)//2  # +1 -1 
        minimum = np.min(spike[:, channel_idx])
        maximum = np.max(spike[:, channel_idx])
        mean = (np.mean(spike[:, :channel_idx]) + np.mean(spike[:, channel_idx+1:]))/2.

        amplitude = maximum - minimum 
        snr = maximum / mean

        return snr


    def preprocess_with_label(self, x, y):


        ## generate noise also 

        self.raw_x = RawArray('f', x.shape[0] * x.shape[1])
        self.x = np.frombuffer(self.raw_x, dtype=np.float32).reshape(x.shape)

        self.raw_y = RawArray('f', y.shape[0] * y.shape[1])
        self.y = np.frombuffer(self.raw_y, dtype=np.float32).reshape(y.shape)


        self.x = x
        self.y = y

        #np.copyto(self.x, x)
        #np.copyto(self.y, y)

        processes = 5#multiprocessing.cpu_count()


        #self.create_blank_samples()

        self.setup_neuron_channels(y)


        self._nbefore_waveform = 45#int(self.config["waveform_before_ms"] * self.config["freq"] // 1000)
        self._nafter_waveform = 60#int(self.config["waveform_after_ms"] * self.config["freq"] // 1000 )

        for i in range(len(y)):
            self.compute_channel_for_neuron(i, max_element=None)

        for i in range(len(y)):
            #self.compute_channel_for_neuron(i, max_element=None)
            self.process_neuron_with_activations(i)

        print("- Successfully generated all samples and labels")
        #return originals, snaps, labels

    def create_anchors(self, anchor_width=5):

        ## this is a very simple anchor system 
        ## every point of the original shape will have an anchor, where the point is the center of the particular anchor 
        
        anchors = [] 
        for i in range(self.config["snap_timespan"]):
            ch = []
            for j in range(self.config["channel_num"]):

                x1 = max(j - anchor_width//2, 0)
                y1 = max(i-self.config["spike_timespan"]//3, 0)

                x2 = min(j + anchor_width//2, self.config["channel_num"]-1)
                y2 = min(i+self.config["spike_timespan"], self.config["snap_timespan"]-1)

                x1 = x1 #/ self.config["channel_num"] 
                x2 = x2 #/ self.config["channel_num"] 

                y1 = y1 #/ self.config["snap_timespan"]
                y2 = y2 #/ self.config["snap_timespan"]

                ch.append([x1,y1,x2,y2,0])

            anchors.append(ch)

        return np.asarray(anchors) # with shape (timespan, channels, 4)

    
    def process_neuron_with_activations(self, n_id, sample_size=7000):

        """
        We will generate for the given n_id cluster with ids from 1 to N for the clusters, whilsts
        for the background, it will have a value of 0 (zero)

        """
        
        x = self.x 
        y = self.y 

        n_act = y[n_id]
        originals = []
        snaps = []
        labels = []

        waveforms = []

        spike_timespan = self.config["spike_timespan"]
        snap_time = self.config["snap_timespan"]

        assert(spike_timespan < snap_time), "Too large spike timespan: Does not fit into the snap_time"

        file_path = self.get_file_path(n_id)

        if not self.overwrite_tfrecords and os.path.isfile(file_path):
            return  

        print("---- Generating for neuron with id: "+str(n_id)+"/"+str(len(self.y))+"... ")

        dataset_spike_pos_bias = 0

        if self.config["dataset_name"] == "STATIC_8X_A_2A":
            dataset_spike_pos_bias = 10

        for act in n_act:

            spike_templates = []

            anchors = self.create_anchors()
            spike_labels = []

            original_snap = []

            if act == 0:
                continue

            lower_bound = max((act-sample_size//2), 0)
            center = act - lower_bound
            upper_bound = min(lower_bound + sample_size, len(x)-1) # attention!! in special cases the target spike won't be at the center
            
            
            sample = x[lower_bound:upper_bound]

            gc.collect()

            try:
                sample = np.array(butter_bandpass_filter(sample, 300, 3000, self.config["freq"], 5, axis=0))
            
            except Exception:
                continue

            ## WAVEFORM EXTRACTION SEGMENT

            if len(sample) >= center+self._nafter_waveform and center >= self._nbefore_waveform:
                waveforms.append(sample[center - self._nbefore_waveform:center+self._nafter_waveform, self.neuron_channels[n_id]])

            ## WAVEFORM EXTRACTION SEGMENT END 

            #""" WE WILL NEED THIS 
            # Get all the other spikes which overlap in the current segment
            for i in range(len(y)):
                if i == n_id:
                    continue 
                    
                n_arr = y[i]
                
                pos = n_arr[n_arr>=act-snap_time//2]
                
                pos = pos[pos<act+snap_time//2]
                
                    
                for k in range(len(pos)): 
                    dt =  pos[k] - act 
                    
                    #anchor_idx = (dt) * self.neuron_channels[i]
                    
                    if dt+snap_time//2 < self.config["snap_timespan"]:
                        anchors[min(dt+snap_time//2+dataset_spike_pos_bias, len(anchors)-1), self.neuron_channels[i], 4] = i+1

                    spike_labels.append(anchors[min(dt+snap_time//2+dataset_spike_pos_bias, len(anchors)-1), self.neuron_channels[i]])

            #"""    
       
            sample = sample[center-snap_time//2:center+snap_time//2]
            
            #anchor_idx = (snap_time//2-1) * self.neuron_channels[n_id]

            anchors[snap_time//2-1+dataset_spike_pos_bias, self.neuron_channels[n_id], 4] = n_id+1
            spike_labels.append(anchors[min(snap_time//2-1+dataset_spike_pos_bias, len(anchors)-1), self.neuron_channels[n_id]])

            spike_labels = np.array(anchors, dtype=np.float32)
            spike_labels = np.reshape(spike_labels, (-1, 5))

            original_snap = np.asarray(sample, dtype=np.float32)

            originals.append(original_snap)
            labels.append(spike_labels)

            gc.collect()

        ## WAVEFORM MANAGER
        waveforms = np.array(waveforms)
        waveforms = np.mean(waveforms, axis=0)

        waveform_savefile = os.path.join(self.config["save_dir"], self.config["save_name"], self.config["default_waveform_dir"], str(n_id)+".npy")

        os.makedirs(os.path.dirname(waveform_savefile), exist_ok=True)

        np.save(waveform_savefile, waveforms)

        ### WAVEFORM END 

        if len(originals[-1]) != len(originals[0]):
            originals = originals[:-1]
            labels = labels[:-1]

        try:
            originals = np.asarray(originals, dtype=np.float32)
            labels = np.asarray(labels, dtype=np.float32)

            assert (originals.shape[0] == labels.shape[0]), "Mismatch of first dimension at preprocessing: originals:"+str(originals.shape)+", labels:"+str(labels.shape)
            
            self.config["label_size"] = labels.shape[-1]

            self.write_tfrecord(file_path, originals, labels)
        
        except Exception as e:
            print("Exception at write tfrecord ")
            print(e)
        #yield n_id, originals, snaps, labels   

    def write_tfrecord(self, fname, template, labels):

        os.makedirs(os.path.dirname(fname), exist_ok=True)


        with tf.io.TFRecordWriter(fname) as f:

            for sample,label in zip(template, labels):
                ## save the gt to f 
                #feature = {'label': _bytes_feature(tf.compat.as_bytes(label.tostring())), 'template': _bytes_feature(tf.compat.as_bytes(sample.tostring()))}

                tf_example = tf.train.Example(features=tf.train.Features(feature={
                    'image/height': dataset_util.int64_feature(self.config["snap_timespan"]),
                    'image/width': dataset_util.int64_feature(self.config["channel_num"]),
                    'image/filename': dataset_util.bytes_feature(tf.compat.as_bytes(fname)),
                    'image/source_id': dataset_util.bytes_feature(tf.compat.as_bytes(self.config["dataset_name"])),
                    'image/encoded':  dataset_util.bytes_feature(tf.compat.as_bytes(sample.tostring())),
                    'image/format': dataset_util.bytes_feature(b"numpy"),
                    'image/object/bbox/xmin': dataset_util.float_list_feature(label[:,0]),
                    'image/object/bbox/ymin': dataset_util.float_list_feature(label[:,1]),
                    'image/object/bbox/xmax': dataset_util.float_list_feature(label[:,2]),
                    'image/object/bbox/ymax': dataset_util.float_list_feature(label[:,3]),
                    'image/object/class/label': dataset_util.float_list_feature(label[:,4]),
                }))

                #example = tf.train.Example(features=tf.train.Features(feature=feature))
                f.write(tf_example.SerializeToString())


    def create_blank_samples(self, num_samples=10000, initial_sample_size=10000, fake_per_sample = 3):

        x = self.x 
        # we are presuming that at this point the self.activations is sorted already
        diff = np.diff(self.activations)

        mask = np.where(np.greater(diff, self.config["snap_timespan"]), 1, 0).astype(np.bool)

        val = np.where(np.greater(diff, self.config["snap_timespan"]), diff//2, 0)

        val_mask = np.zeros_like(self.activations)
        bool_mask = np.zeros_like(self.activations, dtype=np.bool)

        bool_mask[:-1] = mask
        val_mask[:-1] = val

        m_act = self.activations + val_mask

        blank_spaces = m_act[bool_mask]

        fake_snap = []
        fake_labels = []
 
        file_path = self.get_file_path(-1) # -1 is the class of fake, activity-free samples

        for i in range(min(len(blank_spaces), num_samples)):

            fake_act = blank_spaces[i]

            anchors = self.create_anchors()

            lower_bound = max((fake_act-initial_sample_size//2), 0)
            center = fake_act - lower_bound
            upper_bound = min(lower_bound + initial_sample_size, len(x)-1) # attention!! in special cases the target spike won't be at the center
            
            sample = x[lower_bound:upper_bound]

            sample = np.array(butter_bandpass_filter(sample, 300, 3000, self.config["freq"], 5, axis=0))

            sample = sample[center-self.config["snap_timespan"]//2:center+self.config["snap_timespan"]//2]


            ## this should be random 

            fake_x = np.random.default_rng().uniform(low=0, high=self.config["snap_timespan"], size=fake_per_sample).astype(np.int32)
            fake_y = np.random.default_rng().uniform(low=0, high=self.config["channel_num"], size=fake_per_sample).astype(np.int32)

            anchors[fake_x, fake_y, 4] = -1
            spike_labels = anchors[fake_x, fake_y]

            spike_labels = np.array(anchors, dtype=np.float32)
            spike_labels = np.reshape(spike_labels, (-1, 5))

            original_snap = np.asarray(sample, dtype=np.float32)

            fake_snap.append(original_snap)
            fake_labels.append(spike_labels)

        try:
            fake_snap = np.asarray(fake_snap, dtype=np.float32)
            fake_labels = np.asarray(fake_labels, dtype=np.float32)

            assert (fake_snap.shape[0] == fake_labels.shape[0]), "Mismatch of first dimension at preprocessing: fake_snap:"+str(fake_snap.shape)+", fake_labels:"+str(fake_labels.shape)
            
            self.config["label_size"] = fake_labels.shape[-1]

            self.write_tfrecord(file_path, fake_snap, fake_labels)
            

            print("---- Successfully created blank samples", end="\r\n\n")
        except Exception as e:
            print(e)