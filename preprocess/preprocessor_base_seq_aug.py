import os
import sys

import glob
import numpy as np
import xmltodict

import tensorflow as tf
from scipy.signal import butter, lfilter
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import gc
import multiprocessing
from multiprocessing import Pool
from multiprocessing import RawArray

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

    def __init__(self, config_file ="", config={}, overwrite=False):

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
        self.config.setdefault("data_source", "")
        self.config.setdefault("channel_num", 128)
        self.config.setdefault("freq", 20000)
        self.config.setdefault("spike_timespan", 32)
        self.config.setdefault("snap_timespan", 128)
        self.config.setdefault("labeling_length", 15)
        self.config.setdefault("channel_bounding_box", 2)
        self.config.setdefault("overlap_limit", 5) # to remove or include spike from end/beginning of segment 
        #self.config.setdefault("target_dim", 128) # target_dim = S = 128x128

        ## Preprocessing parameters

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

        # this is the end of the timespan where a spike might start, but the whole waveform cannot be seen, so we do not label them
        death_zone = self.config["snap_timespan"] - self.config["overlap_limit"]

        label = np.zeros((self.config["snap_timespan"], self.config["channel_num"]), dtype=np.float32)
        if neuron_idx != -1:
            
            assert timestep >=-self.config["overlap_limit"] and neuron_idx >=0 and timestep<self.config["snap_timespan"] and neuron_idx < len(self.neuron_channels), "Incorrect label values"

            if  timestep >= death_zone:
                return label

            for t in range(self.config["labeling_length"]):
                label[max(min(timestep+t, self.config["snap_timespan"]-1),0), self.neuron_channels[neuron_idx]] = neuron_idx+1
            #label[2, timestep, self.neuron_channels[neuron_idx]] = neuron_idx

            #return [float(neuron_channels[neuron_idx]), float(amplitude),  float(timestep)]

        return label

    def get_file_path(self, n_id):
        return os.path.join(self.config["save_dir"], self.config["save_name"], str(n_id)+"_gt.tfrecord")

    def compute_neuron_channels(self, neurons=[], max_element = 100, sample_size = 10000):

        if len(neurons) == 0:
            neurons = self.neurons

        self.neuron_channels = np.ones((len(neurons),), dtype=np.int32) * -1 
        
        print("- Computing the closest channel for each neuron ... ")
        for n_id in range(len(neurons)):
            print("---- Computing for neuron with id: "+str(n_id), end="\r")
            max_channels = []

            if not self.overwrite_tfrecords and os.path.isfile(self.get_file_path(n_id)):
                continue  

            for i, t_spike in enumerate(neurons[n_id]): 
                if t_spike == 0: # because of the padding
                    continue 

                if len(max_channels) > max_element: # for lowering the processing time 
                    break

                lower_bound = max((t_spike-sample_size//2), 0)

                center = t_spike - lower_bound

                upper_bound = min(lower_bound + sample_size, len(self.raw_data)-1) # attention!! in special cases the target spike won't be at the center

                sample = self.raw_data[lower_bound:upper_bound]

                sample = np.array(butter_bandpass_filter(sample, 300, 3000, self.config["freq"], 5, axis=0))

                snap = sample[center-50:center+50]

                c = np.argmax(np.abs(snap[50]-snap[55]) * np.var(snap, axis=0))
                max_channels.append(c)
            
            y = np.bincount(max_channels)

            self.neuron_channels[n_id] = np.argmax(y)


        print("---- Successfully identified all the channels", end="\r\n\n")
        return self.neuron_channels # for simplicity 


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

    def calc_channel_boundaries(self, channel_idx):

        """
        ATTENTION!!
        Currently the channel boundary calculations are made based on a linear channel boundary, which 
        is good for a 2,4 column electrodes, but for higher number of columns it will give FALSE results

        """
        bounding_box = self.config["channel_bounding_box"]
        ## check if bounding box is small enough to fit into the channel number
        assert(bounding_box*2+1 <= self.config["channel_num"]-1)

        lower_bound = max((channel_idx-bounding_box), 0)
        upper_bound = min((channel_idx+bounding_box+1), self.config["channel_num"])

        selected_channels = list(range(lower_bound, upper_bound))
        # check if it has the requested size 
        # if not, pad with noise from the other side of the 

        if(channel_idx-bounding_box < 0):
            tmp_ch = list(range(upper_bound, int(upper_bound+abs(channel_idx-bounding_box))))
            selected_channels = tmp_ch + selected_channels

        if(channel_idx+bounding_box+1 > self.config["channel_num"]):
            tmp_ch = list(range(lower_bound - 1 - (channel_idx+bounding_box - self.config["channel_num"]), lower_bound))
            selected_channels.extend(tmp_ch)
            #selected_channels = tmp_ch.extend(selected_channels)

        return selected_channels

    def preprocess_with_label(self, x, y):


        ## generate noise also 

        self.raw_x = RawArray('f', x.shape[0] * x.shape[1])
        self.x = np.frombuffer(self.raw_x, dtype=np.float32).reshape(x.shape)

        self.raw_y = RawArray('f', y.shape[0] * y.shape[1])
        self.y = np.frombuffer(self.raw_y, dtype=np.float32).reshape(y.shape)

        #np.copyto(self.x, x)
        #np.copyto(self.y, y)

        processes = 5#multiprocessing.cpu_count()

        neuron_channels = self.compute_neuron_channels(y)



        self.x = x
        self.y = y


        for i in range(len(y)):
            self.process_neuron_with_activations(i)

        print("- Successfully generated all samples and labels")
        #return originals, snaps, labels

    def process_neuron_with_activations(self, n_id, sample_size=10000):

        
        x = self.x 
        y = self.y 

        n_act = y[n_id]
        originals = []
        snaps = []
        labels = []

        spike_timespan = self.config["spike_timespan"]
        snap_time = self.config["snap_timespan"]

        assert(spike_timespan < snap_time), "Too large spike timespan: Does not fit into the snap_time"

        file_path = self.get_file_path(n_id)

        if not self.overwrite_tfrecords and os.path.isfile(file_path):
            return  

        print("---- Generating for neuron with id: "+str(n_id)+"/"+str(len(self.y))+"... ")

        for act in n_act:

            spike_templates = []
            spike_labels = self.get_label_list(-1)

            original_snap = []

            if act == 0:
                continue

            lower_bound = max((act-sample_size//2), 0)
            center = act - lower_bound
            upper_bound = min(lower_bound + sample_size, len(x)-1) # attention!! in special cases the target spike won't be at the center
            
            
            sample = x[lower_bound:upper_bound]

            sample = np.array(butter_bandpass_filter(sample, 300, 3000, self.config["freq"], 5, axis=0))
            
            # Get all the other spikes which overlap in the current segment
            for i in range(len(y)):
                if i == n_id:
                    continue 
                    
                n_arr = y[i]
                
                pos = n_arr[n_arr>=act-snap_time//2 - self.config["overlap_limit"]]
                
                pos = pos[pos<act+snap_time//2]
                
                    
                for k in range(len(pos)): 
                    dt =  pos[k] - act 
                    l = max(0, center + dt - spike_timespan//2 )  
                    u = min(sample_size-1, center + dt + spike_timespan//2 ) 

                    box = self.calc_channel_boundaries(self.neuron_channels[i])

                    surplus_spike = np.array(sample[l:u, box])

                    amplitude = self.calc_SNR(surplus_spike)
                    
                    spike_labels = spike_labels + self.get_label_list(i, amplitude,  dt +snap_time//2)
                    
       
            sample = sample[center-snap_time//2:center+snap_time//2]
            

            spike_labels = spike_labels + self.get_label_list(n_id, timestep=snap_time//2-1)


            spike_labels = np.asarray(spike_labels, dtype=np.float32)
            original_snap = np.asarray(sample, dtype=np.float32)

            originals.append(original_snap)
            labels.append(spike_labels)

        try:
            originals = np.asarray(originals, dtype=np.float32)
            labels = np.asarray(labels, dtype=np.float32)

            assert (originals.shape[0] == labels.shape[0]), "Mismatch of first dimension at preprocessing: originals:"+str(originals.shape)+", labels:"+str(labels.shape)
            
            self.config["label_size"] = labels.shape[-1]

            self.write_tfrecord(file_path, originals, labels)
        
        except Exception as e:
            print(e)
        #yield n_id, originals, snaps, labels   

    def write_tfrecord(self, fname, template, labels):

        os.makedirs(os.path.dirname(fname), exist_ok=True)

        with tf.io.TFRecordWriter(fname) as f:

            for sample,label in zip(template, labels):
                ## save the gt to f 
                feature = {'label': _bytes_feature(tf.compat.as_bytes(label.tostring())), 'template': _bytes_feature(tf.compat.as_bytes(sample.tostring()))}

                example = tf.train.Example(features=tf.train.Features(feature=feature))
                f.write(example.SerializeToString())

        
    def introduce_drift(self, sample, drift_fr = 1, drift_max_amp = 5):

        # sample is of shape [timestep, channel_size]

        for i in range(sample.shape[0]):

            drift_value = np.sin(i / drift_fr) * drift_max_amp