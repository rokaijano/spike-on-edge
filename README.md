# Spike Sorting using edgeTPU 

In this repository you will find the implementation of the spike sorting system described in [Edge computing on TPU for brain implant signal analysis](https://doi.org/10.1016/j.neunet.2023.02.036).

### Abstract 
<details>
	<summary> Click to expand </summary>
	<p>
The ever-increasing number of recording sites of silicon-based probes imposes a great challenge for detecting and evaluating single-unit activities in an accurate and efficient manner. Currently separate solutions are available for high precision offline evaluation and separate solutions for embedded systems where computational resources are more limited. We propose a deep learning-based spike sorting system, that utilizes both unsupervised and supervised paradigms to learn a general feature embedding space and detect neural activity in raw data as well as predict the feature vectors for sorting. The unsupervised component uses contrastive learning to extract features from individual waveforms, while the supervised component is based on the MobileNetV2 architecture. One of the key advantages of our system is that it can be trained on multiple, diverse datasets simultaneously, resulting in greater generalizability than previous deep learning-based models. We demonstrate that the proposed model does not only reaches the accuracy of current state-of-art offline spike sorting methods but has the unique potential to run on edge Tensor Processing Units (TPUs), specialized chips designed for artificial intelligence and edge computing. We compare our model performance with state of art solutions on paired datasets as well as on hybrid recordings as well. The herein demonstrated system paves the way to the integration of deep learning-based spike sorting algorithms into wearable electronic devices, which will be a crucial element of high-end brain-computer interfaces.
	</p>
</details>



### TODOs

 - [ ] **SpikeInterface** integration
 - [ ] Further training the feature extractor on at least 5 datasets
 - [ ] Lowering the overhead of the NMS  
 - [ ] creating *documentation* and '***Getting started***' section 
 
 ## Contributing

We welcome contributions of all kinds, including new features, improved model architectures. To contribute please create an issue. 
 ## Citation 
 

    @article{ROKAI2023,
    title = {Edge computing on TPU for brain implant signal analysis},
    journal = {Neural Networks},
    year = {2023},
    issn = {0893-6080},
    doi = {https://doi.org/10.1016/j.neunet.2023.02.036},
    url = {https://www.sciencedirect.com/science/article/pii/S0893608023001089},
    author = {János Rokai and István Ulbert and Gergely Márton},
    keywords = {Spike sorting, Deep learning, Brain-computer interface, Feature extraction, Edge device, Electrophysiology}
    }


## License

This program is free software; you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation; either version 2 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with this program. If not, see  [http://www.gnu.org/licenses](http://www.gnu.org/licenses).
