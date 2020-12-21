from .base import Base
import numpy as np
import tensorflow as tf
import time
"""
Inspired from 
[1]T. Unterthiner, D. Keysers, S. Gelly, O. Bousquet, and I. Tolstikhin, “Predicting Neural Network Accuracy from Weights,” arXiv:2002.11448 [cs, stat], May 2020, Accessed: Sep. 27, 2020. [Online]. Available: http://arxiv.org/abs/2002.11448.
"""

class Weights(Base):
    def __init__(self, n_batches, batch_size, is_demogen, till_layer=4):
        name = "Weights"
        feature_names = [
            stat_name + "_" + str(layer) + "_" + str(param_indx)  
            for param_indx in range(2)
            for layer in range(till_layer)
            for stat_name in [
                "stats_0",
                "stats_25",
                "stats_50",
                "stats_75",
                "stats_100",
                "variance",
                "mean"
            ]
        ]
        super(Weights, self).__init__(name, feature_names, is_demogen)
        self.n_batches = n_batches
        self.batch_size = batch_size
        self.till_layer = till_layer

    def extract_features(self, model, dataset):
        def extract_summary_features(w, qts=(0, 25, 50, 75, 100)):
            """Extract various statistics from the flat vector w."""
            features = np.percentile(w, qts)
            features = np.append(features, [np.std(w), np.mean(w)])
            return features
            
        start_time = time.time() 
        features = []
        for layer in model.layers:
            layer_features = []
            if hasattr(layer, "weights"):
                for param in layer.weights:
                    if "kernel" in param.name or "bias" in param.name:
                        layer_features.append(extract_summary_features(param.numpy().flatten()))
                if len(layer_features)>0: 
                    features.append(layer_features)
            if len(features) == self.till_layer:
                break
        self.last_runtime = time.time() - start_time
        return np.stack(features).flatten()
