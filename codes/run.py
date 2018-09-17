import read_dataset
import argparse
from Trainer import Trainer
import config
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dir", help="Input data directory path")
    args = parser.parse_args()

    data_manager = read_dataset.DatasetReader()

    train = Trainer()
    train.run_train(
        data_manager=data_manager.read_dataset(args.dir, test=False),
        test_data=data_manager.read_dataset(args.dir, test=True),
        model_save_path=os.path.join(config.MODEL_SAVE_FOLDER, config.MODEL_SAVE_NAME),
        sample_save_path = args.dir
    )
