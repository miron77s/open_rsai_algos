from mrcnn import Config

GREEN_CLASS_NAMES = ['BG', 'green']

class GreeneryConfig(Config):
    NAME = "green"
    IMAGES_PER_GPU = 1
    GPU_COUNT = 1
    NUM_CLASSES = len(GREEN_CLASS_NAMES)
    STEPS_PER_EPOCH = 600
    VALIDATION_STEPS = 20

    LEARNING_RATE = 0.001

    BACKBONE = "resnet101"

    IMAGE_MIN_DIM = 1024
    IMAGE_MAX_DIM = 1024

    RPN_NMS_THRESHOLD = 0.65
    RPN_ANCHOR_SCALES = (32, 64, 128, 256, 512)
    DETECTION_MIN_CONFIDENCE = 0.7
    DETECTION_NMS_THRESHOLD = 0.3

HYDRO_CLASS_NAMES = ['BG', 'hydro']

class HydroConfig(Config):
    NAME = "hydro"

    IMAGES_PER_GPU = 1

    GPU_COUNT = 1

    # Number of classes (including background)
    NUM_CLASSES = len(HYDRO_CLASS_NAMES)

    STEPS_PER_EPOCH = 200
    VALIDATION_STEPS = 20

    LEARNING_RATE = 0.001

    BACKBONE = "resnet101"

    IMAGE_MIN_DIM = 1024
    IMAGE_MAX_DIM = 1024

    RPN_NMS_THRESHOLD = 0.7
    RPN_ANCHOR_SCALES = (32, 64, 128, 256, 512)
    DETECTION_MIN_CONFIDENCE = 0.7
    DETECTION_NMS_THRESHOLD = 0.3