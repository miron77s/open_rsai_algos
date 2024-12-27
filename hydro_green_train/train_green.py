import os

import mrcnn.utils
import mrcnn.config
import mrcnn.model
import mrcnn.visualize

import logging
logging.getLogger().setLevel(logging.ERROR)

from mrcnn.config import Config
from mrcnn import model as modellib, utils

from mrcnn.open_rsai_dataset import CocoDataset, DEFAULT_DATASET_YEAR
import mrcnn.open_rsai_train

############################################################
#  Configurations
############################################################

CLASS_NAMES = ['BG', 'green']

class CocoConfig(Config):
    """Configuration for training on MS COCO.
    Derives from the base Config class and overrides values specific
    to the COCO dataset.
    """
    # Give the configuration a recognizable name
    NAME = "green"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 4

    # Uncomment to train on 8 GPUs (default is 1)
    GPU_COUNT = 1

    # Number of classes (including background)
    NUM_CLASSES = len(CLASS_NAMES)
    
    STEPS_PER_EPOCH = 250
    VALIDATION_STEPS = 25
    
    LEARNING_RATE = 0.001
    
    BACKBONE = "resnet101"
    
    IMAGE_MIN_DIM = 1024
    IMAGE_MAX_DIM = 1024
    
    RPN_NMS_THRESHOLD = 0.7
    RPN_ANCHOR_SCALES = (32, 64, 128, 256, 512)
    DETECTION_MIN_CONFIDENCE = 0.7
    DETECTION_NMS_THRESHOLD = 0.3

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
    config = CocoConfig()
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

    # Training dataset. Use the training set and 35K from the
    # validation set, as as in the Mask RCNN paper.
    dataset_train = CocoDataset()
    image_ids = dataset_train.load_coco(dataset_dir, "train", year=DEFAULT_DATASET_YEAR)
    print('dataset_train:', len(image_ids))
    dataset_train.prepare()

    # Validation dataset
    dataset_val = CocoDataset()
    val_type = "val" if DEFAULT_DATASET_YEAR in '2017' else "minival"
    image_ids = dataset_val.load_coco(dataset_dir, val_type, year=DEFAULT_DATASET_YEAR)
    print('dataset_val:', len(image_ids))
    dataset_val.prepare()

    # Image Augmentation
    # Right/Left flip 50% of the time
    # augmentation = imgaug.augmenters.Fliplr(0.5)

    # *** This training schedule is an example. Update to your needs ***
    def mrcnn.open_rsai_train.fine_tuned ( model, dataset_train, dataset_val, config )