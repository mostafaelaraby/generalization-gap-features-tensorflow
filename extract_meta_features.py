import os
from sys import argv, path
from logger import Logger
# preparing packages used 
# program_dir should be path to ingestion_program provided by PGDL competition
# https://competitions.codalab.org/competitions/25301#learn_the_details-get_starting_kit
program_dir = "ingestion_program"
assert os.path.isdir(program_dir)
path.append(program_dir)
from data_manager import DataManager  # load/save data and get info about them
from tensorflow.python.keras.utils.generic_utils import default
import argparse
from meta_features import Enabled_Features
import tensorflow as tf
import pandas as pd
import json
from tqdm import tqdm
import json


def save_checkpoint(checkpoint, data, path):
    checkpoint["data"] = data
    with open(path, "w") as fout:
        json.dump(checkpoint, fout)


def load_checkpoint(path):
    if not (os.path.isfile(path)):
        return {"data": []}
    with open(path, "r") as content:
        return json.load(content)


def GetRecursiveFiles(folderPath, file_name):
    results = os.listdir(folderPath)
    outFiles = []
    cntFiles = 0
    for file in results:
        if os.path.isdir(os.path.join(folderPath, file)):
            outFiles += GetRecursiveFiles(os.path.join(folderPath, file), file_name)
        elif file == file_name:
            outFiles.append(os.path.join(folderPath, file))
            cntFiles = cntFiles + 1
    return outFiles


def extract_features(config, mid, basename, model, training_data, is_demogen=False):
    features_dict = {"mid": mid, "task": basename}
    for feature in Enabled_Features:
        # init feature
        extractor = feature(config.n_batches, config.batch_size, is_demogen)
        logger.info(
            "Extracting {} features {}".format(
                len(extractor.get_feature_names()), extractor.name
            )
        )
        features_set = extractor.extract_features(model, training_data)
        features_dict.update(
            dict(zip(extractor.get_feature_names(), features_set.flatten()))
        )
        logger.info(
            "Time to extract {} is {} second".format(
                extractor.name, extractor.get_run_time()
            )
        )
    return features_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pgdl-parent-path", "-pg", help="PGDL dataset path", required=True
    )
    parser.add_argument(
        "--n-batches", "-nb", help="Number of batches to consider", default=-1, type=int
    )
    parser.add_argument("--batch-size", "-bs", help="Batch size", default=16, type=int)
    parser.add_argument("--save-every", "-se", help="save every", default=10, type=int)
    parser.add_argument(
        "--output-path", "-op", help="Output csv dir", default="../", type=str
    )
    config = parser.parse_args()
    logger = Logger.__call__(config.output_path).get_logger()

    # checking if we have a saed checkpoint to continue extraction
    ckpt_path = os.path.join(config.output_path, "ckpt_features.json")
    checkpoint = load_checkpoint(ckpt_path)
    if len(checkpoint["data"]) != 0:
        logger.info(
            "Continued from Checkpoint having {} models".format(len(checkpoint["data"]))
        )
    models_features = checkpoint["data"]
    ## Parsing PGDL dataset
    assert os.path.isdir(config.pgdl_parent_path)
    # change input dir to compensate for the single file unzipping
    reference_dir = os.path.join(config.pgdl_parent_path, "reference_data")

    input_dir = os.path.join(config.pgdl_parent_path, "input_data")
    datanames = os.listdir(input_dir)
    total_files = GetRecursiveFiles(input_dir, "weights.hdf5")
    pbar = tqdm(total=len(total_files))
    for basename in datanames:
        data_manager = DataManager(basename, input_dir)
        # loading dataset
        training_data = data_manager.load_training_data()
        model_specs_file = os.path.join(reference_dir, basename, "model_configs.json")
        with open(model_specs_file, "r") as f:
            model_specs = json.load(f)
        # measure time to extract features
        for mid in data_manager.model_ids:
            ckpt_key = str(mid) + "_" + basename
            pbar.update(1)
            if ckpt_key in checkpoint:
                continue
            logger.info("Started Extracting features from model {}".format(mid))
            tf.keras.backend.clear_session()
            model = data_manager.load_model(mid)
            mid = mid.replace("model_", "")
            features_dict = extract_features(
                config, mid, basename, model, training_data
            )
            # the target is the generalization gap
            features_dict["target"] = (
                model_specs[mid]["metrics"]["train_acc"]
                - model_specs[mid]["metrics"]["test_acc"]
            )
            models_features.append(features_dict)
            checkpoint[ckpt_key] = 0
            if len(models_features) % config.save_every == 0:
                save_checkpoint(checkpoint, models_features, ckpt_path)
    logger.info("Now creating pandas data frame from features for model training later")

    output_path = os.path.join(config.output_path, "features.csv")
    df_final = pd.DataFrame.from_dict(models_features)
    del models_features
    df_final.to_csv(output_path, index=False)

