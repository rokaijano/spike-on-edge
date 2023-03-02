
import tensorflow as tf
import numpy as np
import os 
from waveform_data import WaveFormDataset

def get_fixed_samples(dataset_name, dataset_path, encoder, min_items=50, save_centroids=False):

    full_path = os.path.join(dataset_path, dataset_name) 
    datasets = tf.constant([dataset_name])

    dset_mean = WaveFormDataset([full_path], batch_size=128, with_batch=True)
    #dset_mean.load_waveforms()
    g = dset_mean()

    t_dset = [dset_mean._waveforms[idx] for idx in dset_mean._waveform_lookup.lookup(datasets) if idx < len(dset_mean._waveforms)]
    dset = np.array(t_dset)[0]

    template_cluster_mean = []

        
    w_min = tf.math.reduce_min(dset, keepdims=True, axis=-1)
    w_max = tf.math.reduce_max(dset, keepdims=True, axis=-1)
    
    dset = (dset - w_min) / (w_max - w_min) # they are not normalized originally
    

    cluster_examples = [[] for x in range(len(dset))]
    
    dit = iter(g)
    
    template_res = np.array(encoder(dset))
    
    while min([len(x) for x in cluster_examples]) < min_items: # until every sample is not filled up 
        
        
        s = next(dit)
        sample_batch = s[0]
        target_batch = s[1][:,:,0]
        
        for sample, target in zip(sample_batch, target_batch):      
            idx =  np.where(target == dset, 1, 0)
            idx = np.all(idx, axis=-1)
            
            idx = np.argmax(idx)    
            
            sample = np.expand_dims(sample, axis=0)
            
            if len(cluster_examples[idx]) < min_items:
                cluster_examples[idx].append(encoder(sample))
        
        print(f"Clusters for dataset {dataset_name}: min: {min([len(x) for x in cluster_examples])}, max: {max([len(x) for x in cluster_examples])}, mean: {int(np.mean([len(x) for x in cluster_examples]))} examples", end="\r")

    
    cluster_examples = np.array(cluster_examples) # of shape (cluster, min_items, 1, embed_dim)
    cluster_examples = np.squeeze(cluster_examples, axis=-2)# shape (cluster, min_items, embed_dim)

    
    print()

    if save_centroids:
        from sklearn.cluster import KMeans


        for cluster_samples in enumerate(cluster_examples):
            kmeans = KMeans(
                init="random",
                n_clusters=20,
                n_init=50,
                max_iter=500,
                random_state=42
            )
            kmeans.fit(cluster_examples)
            we = kmeans.cluster_centers_[0]

            path = os.path.join(full_path, "waveform_embeddings")
            os.makedirs(path, exist_ok=True)
            np.save(os.path.join(path, str(j)+".npy"), we)
        
    return template_res, cluster_examples