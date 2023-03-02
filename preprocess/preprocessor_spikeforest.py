import itertools
try:
    from .preprocessor_base import *
except ImportError:
    from preprocessor_base import *

import kachery as ka
from spikeforest2_utils import AutoRecordingExtractor, AutoSortingExtractor

import spikeinterface as si

from spikeinterface import WaveformExtractor, extract_waveforms

import spikeinterface.toolkit as st
import spikeforest as sf

import pdb

class SpikeForestProcessor(Preprocessor):

    def __init__(self, dataset_name, recording_path, label_path, new_spikeforest=True, *args, **kwargs):
        super(SpikeForestProcessor, self).__init__(*args, **kwargs)
        self.config["dataset_name"] = dataset_name
        ka.set_config(fr='default_readonly')

        self.new_spikeforest = new_spikeforest

        if new_spikeforest:
            R = sf.load_spikeforest_recording(study_name=recording_path, recording_name=dataset_name)
        
            self.recording = R.get_recording_extractor()
            self.sorting_true = R.get_sorting_true_extractor()

        else:
            self.recording = AutoRecordingExtractor(recording_path, download=True)
            self.sorting_true = AutoSortingExtractor(label_path)



        self.config["freq"] = self.recording.get_sampling_frequency()
        self.config["channel_num"] = self.recording.get_num_channels()
        self.neuron_channels = [] # closest channel for each neuron
        
        self.config["save_name"] = dataset_name

        """ Not applicable because of the autorecordingextractor nature 
        print(f"Computing waveforms ...")
        folder = 'waveforms_mearec'
        self.recording.dump = self.recording.dump_to_json
        self.sorting_true.dump = self.sorting_true.dump_to_json

        def get_num_segments():
            return len(self.recording.get_traces()[1])

        def is_filtered():
            return True

        self.recording.get_num_segments = get_num_segments
        self.recording.is_filtered = is_filtered
        self.sorting_true.get_num_segments = get_num_segments
        self.sorting_true.get_sampling_frequency = self.recording.get_sampling_frequency

        #filtered_recording = st.bandpass_filter(self.recording)
        we = WaveformExtractor.create(self.recording, self.sorting_true, folder, remove_if_exists=True)

        we.set_params(ms_before=3., ms_after=4., max_spikes_per_unit=1000)

        we.run_extract_waveforms(n_jobs=11, chunk_size=30000, progress_bar=True)

        die()        
        """

    def load_data_and_labels(self, filename=""):
        # should return a pair of numpy arrays of dimensions ( [timestep, channels], [neuron, activations] )
        self.raw_data = np.array(self.recording.get_traces())

        if not self.new_spikeforest:
            self.raw_data = np.rollaxis(self.raw_data, axis=1)
            
        self.load_labels()

        return self.raw_data, self.neurons
   
    def load_labels(self):
        self.neurons = np.array(())
        for unit in self.sorting_true.get_unit_ids():
            spikes = (self.sorting_true.get_unit_spike_train(unit))
            spikes = np.expand_dims(spikes, 0)

            if self.neurons.shape[0] == 0:
                self.neurons = spikes
            else:
                if self.neurons.shape[-1] > spikes.shape[-1]:
                    spikes = np.pad(spikes, ((0, 0), (self.neurons.shape[-1] - spikes.shape[-1], 0)), mode="constant",
                                    constant_values=0)
                elif self.neurons.shape[-1] < spikes.shape[-1]:
                    self.neurons = np.pad(self.neurons, ((0, 0), (spikes.shape[-1] - self.neurons.shape[-1], 0)),
                                          mode="constant", constant_values=0)

                self.neurons = np.concatenate((self.neurons, spikes), axis=0)
        self.activations = list(itertools.chain.from_iterable((self.neurons)))
        self.activations = np.array(self.activations)
        self.activations = np.sort(self.activations)
        self.loaded_labels = True
        self.neurons = np.asarray(self.neurons)

        
        #we = WaveformExtractor.create(recording, sorting, folder)


