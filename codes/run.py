import read_dataset
# import argparse
from Trainer import Trainer
import config
import os

if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument("dir", help="Input data directory path")
    # args = parser.parse_args()

    train = Trainer()
    train.run_train(
        data_manager=read_dataset.DatasetReader('/disk1/hyeon/UCF-101/ucfTrainTestlist'),
        model_save_path=os.path.join(config.MODEL_SAVE_FOLDER, config.MODEL_SAVE_NAME)
    )
