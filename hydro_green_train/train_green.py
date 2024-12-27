import os

import mrcnn.utils
import mrcnn.config
import mrcnn.model
import mrcnn.visualize
from mrcnn import model as modellib, utils

import logging
logging.getLogger().setLevel(logging.ERROR)

from mrcnn.open_rsai_config import GreeneryTrainConfig
from mrcnn.open_rsai_dataset import CocoDataset, DEFAULT_DATASET_YEAR
import mrcnn.open_rsai_train


############################################################
#  Training
############################################################

if __name__ == '__main__':
    from sys import argv

    if len(argv) != 3 :
        raise Exception("\nUsage:\n" + f"\t{argv[0]} [dataset-dir] [init-weights]")
    
    dataset_dir = argv[1]
    model_path = argv[2]

    # Configurations
    config = GreeneryTrainConfig()
    config.display()

    # Create model
    model = modellib.MaskRCNN(
        mode="training", 
        config=config,
        model_dir=os.getcwd()
    )

    # Load weights
    print("Loading weights ", model_path)
    model.load_weights(
        model_path, 
        by_name=True, 
        exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",  "mrcnn_bbox", "mrcnn_mask"]
    )

    dataset_train = CocoDataset()
    image_ids = dataset_train.load_coco(dataset_dir, "train", year=DEFAULT_DATASET_YEAR)
    print('dataset_train:', len(image_ids))
    dataset_train.prepare()

    dataset_val = CocoDataset()
    val_type = "val" if DEFAULT_DATASET_YEAR in '2017' else "minival"
    image_ids = dataset_val.load_coco(dataset_dir, val_type, year=DEFAULT_DATASET_YEAR)
    print('dataset_val:', len(image_ids))
    dataset_val.prepare()

    mrcnn.open_rsai_train.fine_tuned ( model, dataset_train, dataset_val, config )