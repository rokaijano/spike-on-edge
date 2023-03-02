import os
import sys

#import cv2
import glob
import numpy as np
import xmltodict
import tensorflow as tf

from pathlib import Path
import time 

"""
try:
    from preprocessor_spikeforest import SpikeForestProcessor
except:
    from .preprocessor_spikeforest import SpikeForestProcessor
#"""
#"""

if __name__ == "__main__":

    from preprocessor_spikeforest import SpikeForestProcessor
    from datasets import *

else:

    from .preprocessor_spikeforest import SpikeForestProcessor
    from .datasets import *

#"""

OVERWRITE = False

def preprocess_and_filter(data_dir, config_file="data_config.xml", data_processor = SpikeForestProcessor, train_dict = None, test_dict = None, config={}):


    kampff_train_data = {}
    kampff_test_data = {}
    fiath_train_data = {}
    fiath_test_data = {}
    spikeforest_train_data = {}
    spikeforest_test_data = {}


    ## SPIKEFOREST DATA INPUTS
    #"""
    spikeforest_train_data = {
        SpikeForestProcessor:
            [
                #hybrid_static_siprobe_64C_600_S11,
                #hybrid_static_siprobe_64C_600_S12,
                #REC_32C_600S_11,
                #REC_32C_600S_21,
                #REC_32C_600S_31,
                #REC_32C_600S_32,
                #BOYDEN_32_1103_1_1,
                #YGER_64_20170622
                #BOYDEN_32_509_1_1,
                #BOYDEN_32_419_1_7, 
                #YGER_64_20170621_patch1
                #YGER_64_20170622_patch2
            ]
    }


    all_train_files = []
    all_test_files = []

    if train_dict is None:
        train_dict = {**kampff_train_data, **fiath_train_data, **spikeforest_train_data}
    
    for data_processor, train_files in train_dict.items():
        all_train_files.extend(train_files)
        for f_id, fname in enumerate(train_files):
            if data_processor == SpikeForestProcessor:

                preloaded_ch = []

                if len(fname) >= 5:
                    preloaded_ch = fname[4]

                dp = data_processor(fname[1], fname[2], fname[3], preloaded_channels=preloaded_ch, overwrite=OVERWRITE)  
                dp.run()


    
    print(" --- Config file was written succesfully to "+ config_file) 

    sys.stdout.flush()
   

def main_func():
    data_dir = {
        SpikeForestProcessor : str(Path('data').absolute())      
    }

    t1 = time.time()

    preprocess_and_filter(data_dir)
    print("Finished under >>> ", time.time()-t1)


if __name__ == "__main__":
    main_func()