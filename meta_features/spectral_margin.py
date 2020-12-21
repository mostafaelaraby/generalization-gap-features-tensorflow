from .base import Base
import numpy as np
import numpy.linalg as LA
import tensorflow as tf
import time
"""
Inspired from 
[1]P. Bartlett, D. J. Foster, and M. Telgarsky, “Spectrally-normalized margin bounds for neural networks,” arXiv:1706.08498 [cs, stat], Dec. 2017, Accessed: Oct. 21, 2020. [Online]. Available: http://arxiv.org/abs/1706.08498.
"""

class SpectralMargin(Base):
    def __init__(self, n_batches, batch_size, is_demogen):
        name = "SpectralMargin"
        feature_names = [
            stat_name   
            for stat_name in [
                "stats_0",
                "stats_25",
                "stats_50",
                "stats_75",
                "stats_100",
                "mean",
                "variance"
            ]
        ]
        super(SpectralMargin, self).__init__(name, feature_names, is_demogen)
        self.n_batches = n_batches
        self.batch_size = batch_size

    def extract_features(self, model, dataset):
        @tf.function()
        def compute_margin(x, y):
            logits = model(x)
            num_classes = logits.get_shape().as_list()[1]
            batch_size = tf.shape(logits)[0]
            bs_lin = tf.range(0, batch_size)
            indices_true = tf.stop_gradient(tf.transpose(tf.stack([bs_lin, y])))
            values_true = tf.gather_nd(logits, indices_true)
            values, indices = tf.nn.top_k(logits, k=2)
            # indicator if the highest class matches the ground truth
            true_match_float = tf.cast(
                tf.equal(indices[:, 0], y), dtype=tf.float32
            )
            values_c = values[:, 1] * true_match_float + values[:, 0] * (
                1 - true_match_float
            ) 
            margin = values_true - values_c
            return margin
        
        def get_weights():
            weights = []
            for layer in model.layers:
                if hasattr(layer, "weights"):
                    layer_weights = []
                    for param in layer.weights:
                        if "kernel" in param.name or "bias" in param.name:
                            if "bias" not in param.name:
                                layer_weights.append(tf.reshape(param, (-1, param.shape[-1])))
                            else:
                                layer_weights.append(tf.reshape(param, (1, -1)))
                    if len(layer_weights)>0:
                        weights.append(tf.concat(layer_weights,axis=0))
            return weights

            
        start_time = time.time() 
        # the method itself
        data_batches = dataset.batch(self.batch_size, drop_remainder=True)
        margins = []
        inputs = []
        for i, data in enumerate(data_batches):
            x, y = self.get_input_target(data)
            inputs.append(x)
            # now computing margins
            margin  = compute_margin(x, y)
            margins.append(margin)
            if i == self.n_batches:
                break
        X_fro_norm = np.sqrt(sum([x.numpy().flat[:] ** 2 for x in inputs]).sum())
        # now computing the Lipshitz constant
        # First getting weight matrix
        A = get_weights()
        L2norms = [LA.norm(a.numpy(), ord=2) for a in A]
        L1norms = [LA.norm(a.numpy().flat[:], ord=1) for a in A]
        T = sum(l1**(2/3) / l2**(2/3) for l1, l2 in zip(L1norms, L2norms))
        S = np.prod(L2norms)
        R = T**(3/2) * S
        n = len(margins)
        margin_dist = margins / (R * X_fro_norm / n )
        margin_dist = margin_dist.flatten()
        quartiles = np.percentile(margin_dist, [0, 25, 50, 75, 100])
        mean = np.mean(margin_dist)
        variance = np.std(margin_dist)
        signature = np.append(quartiles, [mean, variance])
        self.last_runtime = time.time() - start_time
        return signature
