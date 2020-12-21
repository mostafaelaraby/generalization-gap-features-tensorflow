from .base import Base
import numpy as np
import tensorflow as tf
import time


class MarginDistribution(Base):
    def __init__(self, n_batches, batch_size, is_demogen, dist_norm=2, epsilon=1e-6, till_layer=4, to_logspace=True):
        name = "Margin Distribution"
        self.dist_norm = dist_norm
        self.epsilon = epsilon
        self.till_layer = till_layer  # Recommended 4 as in the paper considering only first 4 layers input + 3 hidden layers set to -1 to compute on all layers
        self.batch_size = batch_size
        self.n_batches = n_batches  # -1 to use all data
        self.to_logspace = to_logspace # to move features to logspace
        feature_names = [
            stat_name + "_" + str(layer) 
            for layer in range(till_layer)
            for stat_name in [
                "stats_5",
                "stats_25",
                "stats_50",
                "stats_75",
                "stats_95",
                "upper_fence",
                "lower_fence",
            ]
        ]
        super(MarginDistribution, self).__init__(name, feature_names, is_demogen)

    def extract_features(self, model, dataset):
        @tf.function()
        def compute_margin(inputs, labels):
            layer_activations_dict = {}
            x = inputs
            for i, l in enumerate(model.layers):
                x = l(x)
                layer_activations_dict[i] = x
            del x
            with tf.GradientTape(persistent=True) as tape:
                logits = model(inputs, tape=tape)
            num_classes = logits.get_shape().as_list()[1]
            batch_size = tf.shape(logits)[0]
            bs_lin = tf.range(0, batch_size)
            indices_true = tf.stop_gradient(tf.transpose(tf.stack([bs_lin, labels])))
            values_true = tf.gather_nd(logits, indices_true)
            values, indices = tf.nn.top_k(logits, k=2)
            # indicator if the highest class matches the ground truth
            true_match_float = tf.cast(
                tf.equal(indices[:, 0], labels), dtype=tf.float32
            )
            # if zero match the true class then we take the next class, otherwise we use
            # the highest class
            values_c = values[:, 1] * true_match_float + values[:, 0] * (
                1 - true_match_float
            )
            true_match = tf.cast(true_match_float, dtype=tf.int32)
            indices_c = indices[:, 1] * true_match + indices[:, 0] * (1 - true_match)
            grad_ys = tf.one_hot(labels, num_classes)
            grad_ys -= tf.one_hot(indices_c, num_classes)
            # numerator of the distance
            # TODO use only positive value not misclassified data points
            #  For margin distribution, we only consider distances with
            # positive sign (we ignore all misclassified training points). Such design choice facilitates our empirical
            # analysis when we transform our features
            numerator = values_true - values_c

            dct = {}
            layer_activations = []
            for i, l in enumerate(model.layers):
                if self.till_layer != -1 and len(dct) == self.till_layer:
                    break
                try:
                    layer_dims = l._last_seen_input.shape.rank
                    gradient = tape.gradient(logits, l._last_seen_input, grad_ys)
                    if self.dist_norm == 0:  # l infinity
                        g_norm = self.epsilon + tf.reduce_max(
                            tf.abs(gradient), axis=np.arange(1, layer_dims)
                        )
                    elif self.dist_norm == 1:
                        g_norm = self.epsilon + tf.reduce_sum(
                            tf.abs(gradient), axis=np.arange(1, layer_dims)
                        )
                    elif self.dist_norm == 2:
                        g_norm = tf.sqrt(
                            self.epsilon
                            + tf.reduce_sum(
                                gradient * gradient, axis=np.arange(1, layer_dims)
                            )
                        )
                    else:
                        raise ValueError("only norms supported are 1, 2, and infinity")

                    dct[i] = numerator / g_norm
                    layer_activations.append(
                        tf.reshape(layer_activations_dict[i], (batch_size, -1))
                    )
                except AttributeError:  # no _last_seen_input, layer not wrapped (ex: flatten)
                    dct[i] = None
            return dct, layer_activations

        start_time = time.time() 
        data_batches = dataset.batch(self.batch_size, drop_remainder=True)
        per_layer_dict = {}
        for i, data in enumerate(data_batches):
            x, y  = self.get_input_target(data)
            margin_dist, layer_activations = compute_margin(x, y)
            all_activations = np.concatenate(
                [np.squeeze(activation.numpy()) for activation in layer_activations],
                axis=1,
            )
            response_flat = all_activations.reshape([all_activations.shape[0], -1])
            response_std = np.std(response_flat, axis=0)
            total_variation = (np.sum(response_std ** 2)) ** 0.5
            # make the margin dist scale invariant by dividing on to total variation
            layers_norm = [v / total_variation for v in margin_dist.values()]
            for layer_indx, layer in enumerate(layers_norm):
                if layer_indx not in per_layer_dict:
                    per_layer_dict[layer_indx] = []
                per_layer_dict[layer_indx].append(layer)
            if i == self.n_batches:
                break
        # computing statistical signature
        stats_signature = []
        for layer in per_layer_dict:
            layer_norm_margin = np.concatenate(per_layer_dict[layer], axis=0)
            quartiles = np.percentile(layer_norm_margin, [5, 25, 50, 75, 95])
            inter_quartile = quartiles[-2] - quartiles[1]  # Q3 - Q1
            upper_fence = quartiles[-2] + 1.5 * inter_quartile
            lower_fence = quartiles[1] - 1.5 * inter_quartile
            signature = np.append(quartiles, [upper_fence, lower_fence])
            stats_signature.append(signature)
        stats_signature = np.concatenate(stats_signature)
        if self.to_logspace:
            stats_signature = np.log(np.abs(stats_signature))
        self.last_runtime = time.time() - start_time
        return stats_signature
