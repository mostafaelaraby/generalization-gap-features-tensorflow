import tensorflow as tf
class Base:
    def __init__(self, name, feature_names,is_demogen=False):
        self.name = name
        self.feature_names = [feature_name +"_" + name for feature_name in feature_names] 
        self.last_runtime = -1
        self.is_demogen = is_demogen

    def get_name(self):
        return self.name

    def get_feature_names(self):
        return self.feature_names

    def extract_features(self, model, dataset):
        pass

    def get_run_time(self):
        return self.last_runtime
    
    def get_input_target(self, data): 
        if self.is_demogen:
            # this is for demogen dataset only working on cifar-10
            return data["inputs"], tf.squeeze(tf.cast(data["targets"], tf.int32))
        return data
