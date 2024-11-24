import os

from osgeo import gdal, ogr, osr

import mrcnn.utils
import mrcnn.config
import mrcnn.model
import mrcnn.visualize

import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import json
import datetime
from pathlib import Path
import re
import math
from numpy import zeros

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as maskUtils

from scipy import ndimage

import logging
logging.getLogger().setLevel(logging.ERROR)

# # Import Mask RCNN
# sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import model as modellib, utils

DEFAULT_DATASET_YEAR = "2014"

############################################################
#  Configurations
############################################################

CLASS_NAMES = ['BG', 'green']

class CocoConfigImg1(Config):
    """Configuration for training on MS COCO.
    Derives from the base Config class and overrides values specific
    to the COCO dataset.
    """
    # Give the configuration a recognizable name
    NAME = "green"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 1

    # Uncomment to train on 8 GPUs (default is 1)
    GPU_COUNT = 1

    # Number of classes (including background)
    NUM_CLASSES = len(CLASS_NAMES)
    
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

class CocoConfigImg4(CocoConfigImg1):
    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 4
    
############################################################
#  Dataset
############################################################

class CocoDataset(utils.Dataset):
    def load_coco(self, dataset_dir, subset, year=DEFAULT_DATASET_YEAR, last_id=0, class_ids=None,
                  class_map=None, return_coco=False):
        """Load a subset of the COCO dataset.
        dataset_dir: The root directory of the COCO dataset.
        subset: What to load (train, val, minival, valminusminival)
        year: What dataset year to load (2014, 2017) as a string, not an integer
        class_ids: If provided, only loads images that have the given classes.
        class_map: TODO: Not implemented yet. Supports maping classes from
            different datasets to the same class ID.
        return_coco: If True, returns the COCO object.
        """
        
        last_id += 1

        coco = COCO("{}/annotations/instances_{}{}.json".format(dataset_dir, subset, year))
        if subset == "minival" or subset == "valminusminival":
            subset = "val"
        image_dir = "{}/{}{}".format(dataset_dir, subset, year)

        # Load all classes or a subset?
        if not class_ids:
            # All classes
            class_ids = sorted(coco.getCatIds())

        # # All images or a subset?
        # if class_ids:
        #     image_ids = []
        #     for id in class_ids:
        #         image_ids.extend(list(coco.getImgIds(catIds=[id])))
        #     # Remove duplicates
        #     image_ids = list(set(image_ids))
        # else:
        #     # All images
        #     image_ids = list(coco.imgs.keys())
       
        # All images  
        image_ids = list(coco.imgs.keys())

        # Add classes
        for i in class_ids:
            self.add_class("coco", i, coco.loadCats(i)[0]["name"])

        # Add images
        for i in image_ids:
            # print('add_image image_id:', i)
            self.add_image(
                "coco", image_id=i + last_id,
                path=os.path.join(image_dir, coco.imgs[i]['file_name']),
                width=coco.imgs[i]["width"],
                height=coco.imgs[i]["height"],
                annotations=coco.loadAnns(coco.getAnnIds(
                    imgIds=[i], catIds=class_ids, iscrowd=None)))
            
        for i in range(0, len(image_ids)):
            image_ids[i] += last_id
            
        return image_ids # list(range(0, len(image_ids) - 1))
        if return_coco:
            return coco

    def load_image_gt(self, image_id):
        """Load instance masks for the given image.

        Different datasets use different ways to store masks. This
        function converts the different mask format to one format
        in the form of a bitmap [height, width, instances].

        Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a COCO image, delegate to parent class.
        # print('load_image_gt image_id:', image_id)
        image_info = self.image_info[image_id]
        img_path = image_info['path']
        # load the input image, convert it from BGR to RGB channel
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if image_info["source"] != "coco":
            return super(CocoDataset, self).load_mask(image_id)
        
        bbxes = []
        instance_masks = []
        class_ids = []
        annotations = self.image_info[image_id]["annotations"]
        # Build mask of shape [height, width, instance_count] and list
        # of class IDs that correspond to each channel of the mask.
        for annotation in annotations:
            class_id = self.map_source_class_id(
                "coco.{}".format(annotation['category_id']))
            if class_id:
                m = self.annToMask(annotation, image_info["height"],
                                   image_info["width"])
                # Some objects are so small that they're less than 1 pixel area
                # and end up rounded out. Skip those objects.
                if m.max() < 1:
                    continue
                # Is it a crowd? If so, use a negative class ID.
                if annotation['iscrowd']:
                    # Use negative class ID for crowds
                    class_id *= -1
                    # For crowd masks, annToMask() sometimes returns a mask
                    # smaller than the given dimensions. If so, resize it.
                    if m.shape[0] != image_info["height"] or m.shape[1] != image_info["width"]:
                        m = np.ones([image_info["height"], image_info["width"]], dtype=bool)
                instance_masks.append(m)
                bbox = annotation['bbox']
                bbxes.append(bbox)
                # print('bbox:', bbox)
                class_ids.append(class_id)

        # Pack instance masks into an array
        
        boxes = np.empty([len(bbxes), 4])
        for i in range(0, len(bbxes)):
            boxes[i][0] = int(bbxes[i][1])
            boxes[i][1] = int(bbxes[i][0])
            boxes[i][2] = int(bbxes[i][1] + bbxes[i][3])
            boxes[i][3] = int(bbxes[i][0] + bbxes[i][2])
        
        # print('!class_ids', class_ids)
        if class_ids:
            # print('!@!class_ids', class_ids)
            mask = np.stack(instance_masks, axis=2).astype(np.bool)
            class_ids = np.array(class_ids, dtype=np.int32)
            return img_path, image, class_ids, boxes, mask
        else:
            # Call super class to return an empty mask
            mask, class_ids = super(CocoDataset, self).load_mask(image_id)
            return img_path, image, class_ids, boxes, mask
        
    def load_mask(self, image_id):
        """Load instance masks for the given image.

        Different datasets use different ways to store masks. This
        function converts the different mask format to one format
        in the form of a bitmap [height, width, instances].

        Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a COCO image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != "coco":
            return super(CocoDataset, self).load_mask(image_id)
        
        
        instance_masks = []
        class_ids = []
        annotations = self.image_info[image_id]["annotations"]
        # Build mask of shape [height, width, instance_count] and list
        # of class IDs that correspond to each channel of the mask.
        for annotation in annotations:
            class_id = self.map_source_class_id(
                "coco.{}".format(annotation['category_id']))
            if class_id:
                m = self.annToMask(
                    annotation, 
                    image_info["height"],
                    image_info["width"]
                )
                # Some objects are so small that they're less than 1 pixel area
                # and end up rounded out. Skip those objects.
                if m.max() < 1:
                    continue
                # Is it a crowd? If so, use a negative class ID.
                if annotation['iscrowd']:
                    # Use negative class ID for crowds
                    class_id *= -1
                    # For crowd masks, annToMask() sometimes returns a mask
                    # smaller than the given dimensions. If so, resize it.
                    if m.shape[0] != image_info["height"] or m.shape[1] != image_info["width"]:
                        m = np.ones([image_info["height"], image_info["width"]], dtype=bool)
                instance_masks.append(m)
                class_ids.append(class_id)

        # Pack instance masks into an array
        if class_ids:
            mask = np.stack(instance_masks, axis=2).astype(np.bool)
            class_ids = np.array(class_ids, dtype=np.int32)
            return mask, class_ids
        else:
            # Call super class to return an empty mask
            return super(CocoDataset, self).load_mask(image_id)

    def image_reference(self, image_id):
        """Return a link to the image in the COCO Website."""
        info = self.image_info[image_id]
        if info["source"] == "coco":
            return "http://cocodataset.org/#explore?id={}".format(info["id"])
        else:
            super(CocoDataset, self).image_reference(image_id)

    # The following two functions are from pycocotools with a few changes.

    def annToRLE(self, ann, height, width):
        """
        Convert annotation which can be polygons, uncompressed RLE to RLE.
        :return: binary mask (numpy 2D array)
        """
        segm = ann['segmentation']
        if isinstance(segm, list):
            # polygon -- a single object might consist of multiple parts
            # we merge all parts into one mask rle code
            rles = maskUtils.frPyObjects(segm, height, width)
            rle = maskUtils.merge(rles)
        elif isinstance(segm['counts'], list):
            # uncompressed RLE
            rle = maskUtils.frPyObjects(segm, height, width)
        else:
            # rle
            rle = ann['segmentation']
        return rle

    def annToMask(self, ann, height, width):
        """
        Convert annotation which can be polygons, uncompressed RLE, or RLE to binary mask.
        :return: binary mask (numpy 2D array)
        """
        rle = self.annToRLE(ann, height, width)
        m = maskUtils.decode(rle)
        return m


############################################################

def plot_display(data, title, target_dir):
    if len(data) > 0 :
        print('best epoch ', max(data, key=data.get), ' ', title, round(max(data.values()), 6))
        # with open("DETECTION_NMS_THRESHOLD.txt", "a") as myfile:
        #     myfile.write(
        #         'best epoch ' + str(max(data, key=data.get)) + ' ' + 
        #         title + str(round(max(data.values()), 6)) + '\n')
        # Plot the Precision-Recall curve
        # _, ax = plt.subplots(1)
        # ax.set_title(title)
        # ax.set_ylim(0, 1.1)
        # keys = list(data.keys())
        # ax.set_xlim(keys[0] - 1, keys[-1] + 1)
        # _ = ax.plot(data.keys(), data.values())
        # plt.show()
        _, ax = plt.subplots(1)
        ax.set_title(title)
        ax.set_ylim(0, 1.1)
        keys = list(data.keys())
        ax.set_xlim(keys[0] - 1, keys[-1] + 1)
        _ = ax.plot(data.keys(), data.values())
        plt.savefig(
            target_dir + 'log/' + 
            datetime.datetime.now().strftime("%Y_%m_%d__%H_%M_%S") + 
            ' ' + title + ' epoch ' + str(list(data.keys())[-1]) + '.png'
        )
        # plt.show()

def save_statistics_to_file(log_path, ap_data, iou_data):
    with open(log_path + "ap_data.txt", "a") as myfile:
        for key, value in ap_data.items():
            myfile.write(str(key) + ":" + str(value) + '\n')
    with open(log_path + "iou_data.txt", "a") as myfile:
        for key, value in iou_data.items():
            myfile.write(str(key) + ":" + str(value) + '\n')
    
def update_date(data, epoch, data_list):
    if len(data_list) > 0 :
        data[epoch] = sum(data_list) / len(data_list)
        
# Print iterations progress
def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    # Print New Line on Complete
    if iteration == total: 
        print()

def visualize(image, data, colors, save_path):
    mrcnn.visualize.display_instances(
        image=image, 
        boxes=data['rois'], 
        masks=data['masks'], 
        class_ids=data['class_ids'], 
        class_names=CLASS_NAMES, 
        scores=data['scores'],
        figsize =(18,18),
        ax=None,
        show_mask=True,
        show_mask_polygon=True,
        show_bbox=False,
        colors=colors,
        captions=None,
        show_caption=False,
        save_fig_path=save_path,
        filter_classes=None, 
        min_score=None
    )
            
# The standard work-around: first convert to greyscale 
def img_grey(data):
    return Image.fromarray(data * 255, mode='L').convert('1')

# Use .frombytes instead of .fromarray. 
# This is >2x faster than img_grey
def img_frombytes(data):
    
    masks = zeros([1024, 1024], dtype='bool')
    for i in range(0, 1024):
        for j in range(0, 1024):
            s = False
            for k in range(0, data.shape[2]):
                s = s or data[i][j][k]
            masks[i][j] = s
    return img_grey(masks)

def run(config, dataset_dir, target_dir, test_img_list, epoch_list, save_result):
    log_path = target_dir + 'log/'
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    with open(log_path + "ap_data.txt", "a") as myfile:
        myfile.write('########################\n')
        myfile.write(datetime.datetime.now().strftime("%Y_%m_%d__%H_%M_%S") + '\n')
        myfile.write('########################\n')
    with open(log_path + "iou_data.txt", "a") as myfile:
        myfile.write('########################\n')
        myfile.write(datetime.datetime.now().strftime("%Y_%m_%d__%H_%M_%S") + '\n')
        myfile.write('########################\n')
    with open(log_path + "detections.txt", "a") as myfile:
        myfile.write('########################\n')
        myfile.write(datetime.datetime.now().strftime("%Y_%m_%d__%H_%M_%S") + '\n')
        myfile.write('########################\n')
    # Initialize the Mask R-CNN model for inference and then load the weights.
    # This step builds the Keras model architecture.
    model = mrcnn.model.MaskRCNN(
        mode="inference", 
        config=config, 
        model_dir=os.getcwd()
    )

    # Training dataset. Use the training set and 35K from the
    # validation set, as as in the Mask RCNN paper.
    dataset = CocoDataset()
    image_ids = dataset.load_coco(dataset_dir, "train", year="2014") # valminusminival | train
    image_ids = image_ids + dataset.load_coco(dataset_dir, "valminusminival", year="2014", last_id=max(image_ids))
    print('image_ids:', len(image_ids))
    dataset.prepare()
        
    # Validation dataset
    # dataset = CocoDataset()
    # val_type = "minival"
    # image_ids = dataset.load_coco(DEFAULT_DATASET_DIR, val_type, year="2014")
    # dataset.prepare()
 
    weights_dir = target_dir + "mask_rcnn_green_*epoch*.h5"

    ap_data = {}
    bing_ap_data = {}
    google_ap_data = {}
    mapbox_ap_data = {}
    yandex_ap_data = {}

    iou_data = {}
    bing_iou_data = {}
    google_iou_data = {}
    mapbox_iou_data = {}
    yandex_iou_data = {}
    
    for epoch in epoch_list : # range(1, 150) :
        weights_path = weights_dir.replace("*epoch*", str(epoch).zfill(4))
        with open(log_path + "detections.txt", "a") as myfile:
            myfile.write('epoch: ' + str(epoch) + '\n')

        target_path = target_dir
        if not os.path.exists(target_path):
            os.makedirs(target_path)
        target_path = target_path + "result/"
        if not os.path.exists(target_path):
            os.makedirs(target_path)
        target_path = target_path + Path(weights_path).stem + "/"
        if os.path.exists(target_path):
            print('already tested:', target_path)
        else :
            os.makedirs(target_path) 

        # Load the weights into the model.
        # print('weights_path:', weights_path)
        model.load_weights(
            filepath=weights_path, 
            by_name=True
        )

        ap_list = []
        bing_ap_list = []
        google_ap_list = []
        mapbox_ap_list = []
        yandex_ap_list = []
        precisions_list = np.array([])
        recalls_list = np.array([])
    
        iou_list = []
        bing_iou_list = []
        google_iou_list = []
        mapbox_iou_list = []
        yandex_iou_list = []
    
        gt_class_id_list = np.array([])
        class_id_list = np.array([])
        scores_list = np.array([])
        overlaps_list = np.array([])
        colors = []
        for i in range(1, 100):
            colors.append((0.0, 0.0, 1.0))
    
        # test_img_list = range(1, len(image_ids)) # range(1, len(image_ids)): # range(1, 1000):
        image_num = len(test_img_list)
        print('image_num', image_num)
        printProgressBar(0, image_num, prefix = 'Progress:', suffix = '', length = 50)
        for image_number, image_id in enumerate(test_img_list):
            printProgressBar(image_number + 1, image_num, prefix = 'Progress:', suffix = '', length = 50)
            img_path, image, gt_class_ids, gt_bbox, gt_mask = dataset.load_image_gt(image_id)
            with open(log_path + "detections.txt", "a") as myfile:
                myfile.write('img_path: ' + img_path + '\n')
            if len(gt_bbox) == 0 :
                continue
            gt_data = {}
            gt_data['masks'] = gt_mask
            gt_data['rois'] = gt_bbox
            gt_data['class_ids'] = gt_class_ids
            # print('img_path:', img_path)
            # print("gt_class_id", gt_class_id)
        
            ret = model.detect([image], verbose=0)
            # Get the results for the first image.
            r = ret[0]

            save_path = target_path + Path(img_path).stem + ".jpg"
        
            kernel = np.asarray(
                [[False, True, False],
                [True, True, True],
                [False, True, False]]
            )
            for i in range(0, r['masks'].shape[2]) : 
                x1, y1, x2, y2 =  r['rois'][i]
                w = x2 - x1
                h = y2 - y1
                s = w * h
                iterations = 1
                if s < 64 * 64 :
                    iterations = 0
                elif s < 128 * 128 :
                    iterations = 2
                elif s < 256 * 256 :
                    iterations = 7
                elif s < 512 * 512 :
                    iterations = 20
                else :
                    iterations = 42
                if iterations > 0 :
                    r['masks'][:, :, i] = ndimage.binary_dilation(r['masks'][:, :, i], kernel, iterations)
            
            cur_iou_list = []
            cur_s_list = []
            for i in range(0, gt_mask.shape[2]) : 
                x1, y1, x2, y2 =  gt_bbox[i]
                w = x2 - x1
                h = y2 - y1
                s = w * h
                best_s = 0
                max_iou = 0
                best_num_mask = -1
                for j in range(0, r['masks'].shape[2]) : 
                    component1 = gt_mask[:, :, i]
                    component2 = r['masks'][:, :, j]
                    overlap = component1 * component2 # Logical AND
                    union = component1 + component2 # Logical OR
                    iou = overlap.sum() / float(union.sum())
                    if iou > max_iou :
                        max_iou = iou
                        best_s = w * h
                        best_num_mask = j
                if best_num_mask > -1 :
                    _x1, _y1, _x2, _y2 =  r['rois'][best_num_mask]
                    _w = _x2 - _x1
                    _h = _y2 - _y1
                    with open(log_path + "detections.txt", "a") as myfile:
                       myfile.write(
                       'vegetation: ' + str(int(round(max_iou, 2) * 100)) + '% '
                       ' (left_x: ' + str(_x1)+ ' '*(5-len(str(_x1))) + 'top_y: ' + str(_y1) + ' '*(5-len(str(_y1))) +
                       'width:' + str(_w) + ' '*(5-len(str(_w))) + 'height:' + str(_h) + ' '*(5-len(str(_h))) + ')\n')
            
                cur_iou_list.append(max_iou)  #* best_s
                cur_s_list.append(best_s)
 
            iou = float('nan')
            if len(cur_iou_list) > 0 : #and s_sum > 0 :
                iou = sum(cur_iou_list) / len(cur_iou_list) #/ s_sum statistics.median() #
            
            # Visualize the detected objects.
            if save_result == "save" :
                visualize(image, r, colors, save_path)
            # Visualize GT.
            # visualize(image, gt_data, colors, save_path)
    
            #################
            AP, precisions, recalls, overlaps = mrcnn.utils.compute_ap(
                gt_bbox, gt_class_ids, gt_mask,
                r['rois'], r['class_ids'], r['scores'], r['masks']
            )
            value = float('nan') 
            if not math.isnan(AP) and AP != 0: 
                ap_list.append(AP)
                if re.search('Kursk_Bing', Path(img_path).stem) :
                    bing_ap_list.append(AP)
                if re.search('Kursk_Google', Path(img_path).stem) :
                    google_ap_list.append(AP)
                if re.search('Kursk_Mapbox', Path(img_path).stem) :
                    mapbox_ap_list.append(AP)
                if re.search('Kursk_Yandex', Path(img_path).stem) :
                    yandex_ap_list.append(AP)
            
            if not math.isnan(iou) and iou != 0: 
                iou_list.append(iou)
                if re.search('Kursk_Bing', Path(img_path).stem) :
                    bing_iou_list.append(iou)
                if re.search('Kursk_Google', Path(img_path).stem) :
                    google_iou_list.append(iou)
                if re.search('Kursk_Mapbox', Path(img_path).stem) :
                    mapbox_iou_list.append(iou)
                if re.search('Kursk_Yandex', Path(img_path).stem) :
                    yandex_iou_list.append(iou)
            #################
            # print('AP', AP)
            # print('precisions', precisions)
            # print('recalls', recalls)
            # print('overlaps', overlaps)
            # mrcnn.visualize.plot_precision_recall(AP, precisions, recalls)
        
            # if len(overlaps) != 0 :
            #     mrcnn.visualize.plot_overlaps(
            #         gt_class_ids, 
            #         r['class_ids'], 
            #         r['scores'],
            #         overlaps,
            #         CLASS_NAMES
            #     )
            #################
            # np.concatenate((precisions_list, precisions), axis=0)
            # np.concatenate((recalls_list, recalls), axis=0)
        
            # np.concatenate((gt_class_id_list, gt_class_ids), axis=0)
            # np.concatenate((class_id_list, r['class_ids']), axis=0)
            # np.concatenate((scores_list, r['scores']), axis=0)
            #################
            # np.concatenate((overlaps_list, overlaps), axis=0)
    
        #################
        print('###############')
        update_date(ap_data, epoch, ap_list)
        update_date(bing_ap_data, epoch, bing_ap_list)
        update_date(google_ap_data, epoch, google_ap_list) 
        update_date(mapbox_ap_data, epoch, mapbox_ap_list)
        update_date(yandex_ap_data, epoch, yandex_ap_list)
    
        update_date(iou_data, epoch, iou_list) 
        update_date(bing_iou_data, epoch, bing_iou_list) 
        update_date(google_iou_data, epoch, google_iou_list)
        update_date(mapbox_iou_data, epoch, mapbox_iou_list)  
        update_date(yandex_iou_data, epoch, yandex_iou_list)
    
        #################
    
        # mrcnn.visualize.plot_precision_recall(AP, precisions, recalls)
    
        # mrcnn.visualize.plot_overlaps(
        #     gt_class_id_list, 
        #     class_id_list, 
        #     scores_list,
        #     overlaps_list,
        #     CLASS_NAMES
        # )
        #################
    
        plot_display(ap_data, 'mAP', target_dir)
        plot_display(bing_ap_data, 'bing mAP', target_dir)
        plot_display(google_ap_data, 'google mAP', target_dir)
        plot_display(mapbox_ap_data, 'mapbox mAP', target_dir)
        plot_display(yandex_ap_data, 'yandex mAP', target_dir)
        plot_display(iou_data, 'IoU', target_dir)
        plot_display(bing_iou_data, 'bing IoU', target_dir)
        plot_display(google_iou_data, 'google IoU', target_dir)
        plot_display(mapbox_iou_data, 'mapbox IoU', target_dir)
        plot_display(yandex_iou_data, 'yandex IoU', target_dir)

    print('###############')
                        
    save_statistics_to_file(log_path, ap_data, iou_data)

    
def run_new(config, dataset_dir, target_dir, test_img_list, epoch_list, save_result):
    log_path = target_dir + 'log/'
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    with open(log_path + "ap_data.txt", "a") as myfile:
        myfile.write('########################\n')
        myfile.write(datetime.datetime.now().strftime("%Y_%m_%d__%H_%M_%S") + '\n')
        myfile.write('########################\n')
    with open(log_path + "iou_data.txt", "a") as myfile:
        myfile.write('########################\n')
        myfile.write(datetime.datetime.now().strftime("%Y_%m_%d__%H_%M_%S") + '\n')
        myfile.write('########################\n')
    # Initialize the Mask R-CNN model for inference and then load the weights.
    # This step builds the Keras model architecture.
    model = mrcnn.model.MaskRCNN(
        mode="inference", 
        config=config, 
        model_dir=os.getcwd()
    )

    # Training dataset. Use the training set and 35K from the
    # validation set, as as in the Mask RCNN paper.
    dataset = CocoDataset()
    image_ids = dataset.load_coco(dataset_dir, "train", year="2014") # valminusminival | train
    image_ids = image_ids + dataset.load_coco(dataset_dir, "valminusminival", year="2014", last_id=max(image_ids))
    print('image_ids:', len(image_ids))
    dataset.prepare()
        
    # Validation dataset
    # dataset = CocoDataset()
    # val_type = "minival"
    # image_ids = dataset.load_coco(DEFAULT_DATASET_DIR, val_type, year="2014")
    # dataset.prepare()
   
    weights_dir = target_dir + "mask_rcnn_coco_*epoch*.h5"

    ap_data = {}
    bing_ap_data = {}
    google_ap_data = {}
    mapbox_ap_data = {}
    yandex_ap_data = {}

    iou_data = {}
    bing_iou_data = {}
    google_iou_data = {}
    mapbox_iou_data = {}
    yandex_iou_data = {}

    for epoch in epoch_list : # range(1, 150) :
        weights_path = weights_dir.replace("*epoch*", str(epoch).zfill(4))

        target_path = target_dir
        if not os.path.exists(target_path):
            os.makedirs(target_path)
        target_path = target_path + "result/"
        if not os.path.exists(target_path):
            os.makedirs(target_path)
        target_path = target_path + Path(weights_path).stem + "/"
        if os.path.exists(target_path):
            print('already tested:', target_path)
        else :
            os.makedirs(target_path) 

        # Load the weights into the model.
        # print('weights_path:', weights_path)
        model.load_weights(
            filepath=weights_path, 
            by_name=True
        )

        ap_list = []
        bing_ap_list = []
        google_ap_list = []
        mapbox_ap_list = []
        yandex_ap_list = []
        precisions_list = np.array([])
        recalls_list = np.array([])
    
        iou_list = []
        bing_iou_list = []
        google_iou_list = []
        mapbox_iou_list = []
        yandex_iou_list = []
    
        gt_class_id_list = np.array([])
        class_id_list = np.array([])
        scores_list = np.array([])
        overlaps_list = np.array([])
        colors = []
        for i in range(1, 100):
            colors.append((0.0, 0.0, 1.0))
            
        # test_img_list = range(1, len(image_ids)) # range(1, len(image_ids)): # range(1, 1000):
        image_num = len(test_img_list)
        print('image_num', image_num)
        printProgressBar(0, image_num, prefix = 'Progress:', suffix = '', length = 50)
        for image_number in range(0, image_num, 4) :
            printProgressBar(image_number + 4, image_num, prefix = 'Progress:', suffix = '', length = 50)
            image_id = test_img_list[image_number]
            img_path, image, gt_class_ids, gt_bbox, gt_mask = dataset.load_image_gt(test_img_list[image_number])
            _, image1, _, _, _ = dataset.load_image_gt(test_img_list[image_number + 1])
            _, image2, _, _, _ = dataset.load_image_gt(test_img_list[image_number + 2])
            _, image3, _, _, _ = dataset.load_image_gt(test_img_list[image_number + 3])
            if len(gt_bbox) == 0 :
                continue
            gt_data = {}
            gt_data['masks'] = gt_mask
            gt_data['rois'] = gt_bbox
            gt_data['class_ids'] = gt_class_ids
            # print('img_path:', img_path)
            # print("gt_class_id", gt_class_id)
            
            ret = model.detect([image, image1, image2, image3], verbose=0)

            save_path = target_path + Path(img_path).stem + ".jpg"
        
            kernel = np.asarray(
                [[True, True, True],
                [True, True, True],
                [True, True, True]]
            )
            
            for n in range(0, 4) :
                for i in range(0, ret[n]['masks'].shape[2]) : 
                    x1, y1, x2, y2 =  ret[n]['rois'][i]
                    w = x2 - x1
                    h = y2 - y1
                    s = w * h
                    iterations = 1
                    if s < 64 * 64 :
                        iterations = 0
                    elif s < 128 * 128 :
                        iterations = 2
                    elif s < 256 * 256 :
                        iterations = 7
                    elif s < 512 * 512 :
                        iterations = 20
                    else :
                        iterations = 42
                    if iterations > 0 :
                        ret[n]['masks'][:, :, i] = ndimage.binary_dilation(ret[n]['masks'][:, :, i], kernel, iterations)
            
            sum_mask = zeros((1024, 1024), np.int)
            for n in range(0, 4) :
                for i in range(0, ret[n]['masks'].shape[2]) : 
                    sum_mask += ret[n]['masks'][:, :, i].astype(int)
            binary_mask = sum_mask > 3
            binary_mask = binary_mask * 255
            numpy_array = binary_mask.astype(np.uint8)
            mat_image = np.asarray(numpy_array)
            contours, _ = cv2.findContours(mat_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
            
            # draw Contours
            test_img = image
            cv2.drawContours(test_img, contours, -1, (0, 255, 0), 3)
            if save_result == "save" :
                cv2.imwrite(save_path, test_img)
            
            masks_list = []
            for i, c in enumerate(contours):
                mask = zeros((1024, 1024), np.int)
                cv2.fillPoly(mask, pts=[c], color=(255))
                cv2.drawContours(mask, c, -1, (255),thickness=cv2.FILLED)
                save_mask_path = target_path + Path(img_path).stem + '_' + str(image_number) + '_' + str(i) + '.png'
                if save_result == "save" :
                    cv2.imwrite(save_mask_path, mask)
                masks_list.append(mask)   
                
            cur_iou_list = []
            cur_s_list = []
            for i in range(0, gt_mask.shape[2]) : 
                x1, y1, x2, y2 =  gt_bbox[i]
                w = x2 - x1
                h = y2 - y1
                s = w * h
                # if s < 0 or s > 64 * 64 :
                #     continue
                best_s = 0
                max_iou = 0
                for mask in masks_list : 
                    component1 = gt_mask[:, :, i]
                    component2 = mask
                    overlap = component1 * component2 # Logical AND
                    union = component1 + component2 # Logical OR
                    iou = overlap.sum() / float(union.sum())
                    if iou > max_iou :
                        max_iou = iou
                        best_s = w * h
            
                cur_iou_list.append(max_iou)  #* best_s
                cur_s_list.append(best_s)
    
            iou = float('nan')
            if len(cur_iou_list) > 0 : #and s_sum > 0 :
                iou = sum(cur_iou_list) / len(cur_iou_list) #/ s_sum statistics.median() #
            
            # Visualize the detected objects.
            # visualize(image, r, colors, save_path)
            # Visualize GT.
            # visualize(image, gt_data, colors, save_path)
    
            #################
            AP, precisions, recalls, overlaps = mrcnn.utils.compute_ap(
                gt_bbox, gt_class_ids, gt_mask,
                ret[0]['rois'], ret[0]['class_ids'], ret[0]['scores'], ret[0]['masks']
            )
            value = float('nan') 
            if not math.isnan(AP) and AP != 0: 
                ap_list.append(AP)
                if re.search('Kursk_Bing', Path(img_path).stem) :
                    bing_ap_list.append(AP)
                if re.search('Kursk_Google', Path(img_path).stem) :
                    google_ap_list.append(AP)
                if re.search('Kursk_Mapbox', Path(img_path).stem) :
                    mapbox_ap_list.append(AP)
                if re.search('Kursk_Yandex', Path(img_path).stem) :
                    yandex_ap_list.append(AP)
            
            if not math.isnan(iou) and iou != 0: 
                iou_list.append(iou)
                if re.search('Kursk_Bing', Path(img_path).stem) :
                    bing_iou_list.append(iou)
                if re.search('Kursk_Google', Path(img_path).stem) :
                    google_iou_list.append(iou)
                if re.search('Kursk_Mapbox', Path(img_path).stem) :
                    mapbox_iou_list.append(iou)
                if re.search('Kursk_Yandex', Path(img_path).stem) :
                    yandex_iou_list.append(iou)
            #################
            # print('AP', AP)
            # print('precisions', precisions)
            # print('recalls', recalls)
            # print('overlaps', overlaps)
            # mrcnn.visualize.plot_precision_recall(AP, precisions, recalls)
        
            # if len(overlaps) != 0 :
            #     mrcnn.visualize.plot_overlaps(
            #         gt_class_ids, 
            #         r['class_ids'], 
            #         r['scores'],
            #         overlaps,
            #         CLASS_NAMES
            #     )
            #################
            # np.concatenate((precisions_list, precisions), axis=0)
            # np.concatenate((recalls_list, recalls), axis=0)
        
            # np.concatenate((gt_class_id_list, gt_class_ids), axis=0)
            # np.concatenate((class_id_list, r['class_ids']), axis=0)
            # np.concatenate((scores_list, r['scores']), axis=0)
            #################
            # np.concatenate((overlaps_list, overlaps), axis=0)
    
        #################
        print('###############')
        update_date(ap_data, epoch, ap_list)
        update_date(bing_ap_data, epoch, bing_ap_list)
        update_date(google_ap_data, epoch, google_ap_list) 
        update_date(mapbox_ap_data, epoch, mapbox_ap_list)
        update_date(yandex_ap_data, epoch, yandex_ap_list)
    
        update_date(iou_data, epoch, iou_list) 
        update_date(bing_iou_data, epoch, bing_iou_list) 
        update_date(google_iou_data, epoch, google_iou_list)
        update_date(mapbox_iou_data, epoch, mapbox_iou_list)  
        update_date(yandex_iou_data, epoch, yandex_iou_list)
    
        #################
    
        # mrcnn.visualize.plot_precision_recall(AP, precisions, recalls)
    
        # mrcnn.visualize.plot_overlaps(
        #     gt_class_id_list, 
        #     class_id_list, 
        #     scores_list,
        #     overlaps_list,
        #     CLASS_NAMES
        # )
        #################
    
        plot_display(ap_data, 'mAP', target_dir)
        plot_display(bing_ap_data, 'bing mAP', target_dir)
        plot_display(google_ap_data, 'google mAP', target_dir)
        plot_display(mapbox_ap_data, 'mapbox mAP', target_dir)
        plot_display(yandex_ap_data, 'yandex mAP', target_dir)
        plot_display(iou_data, 'IoU', target_dir)
        plot_display(bing_iou_data, 'bing IoU', target_dir)
        plot_display(google_iou_data, 'google IoU', target_dir)
        plot_display(mapbox_iou_data, 'mapbox IoU', target_dir)
        plot_display(yandex_iou_data, 'yandex IoU', target_dir)

    print('###############')
    save_statistics_to_file(log_path, ap_data, iou_data)
 
test_img_list = [
    445, 1466, 2487, 3508, # 19_10_25
    38, 1059, 2080, 3101, # 19_0_42
    607, 1628, 2649, 3670, # 19_21_26
    285, 1306, 2327, 3348, # 19_5_20
    39, 1060, 2081, 3102, # 19_0_43
    247, 1268, 2289, 3310, # 19_4_30
    256, 1277, 2298, 3319, # 19_4_39
    367, 1388, 2409, 3430, # 19_8_3
    275, 1296, 2317, 3338, # 19_5_8
    198, 1219, 2240, 3261, # 19_3_34
    235, 1256, 2277, 3298, # 19_4_17
    228, 1249, 2270, 3291, # 19_4_10
    223, 1244, 2265, 3286, # 19_4_4
    81, 1102, 2123, 3144, # 19_1_26
    526, 1547, 2568, 3589, # 19_15_5
    580, 1601, 2622, 3643, # 19_19_25
    192, 1213, 2234, 3255, # 19_3_27
    135, 1156, 2177, 3198, # 19_2_23
    447, 1468, 2489, 3510, # 19_10_27
    11, 1032, 2053, 3074, # 19_0_12
    43, 1064, 2085, 3106, # 19_0_47
    335, 1356, 2377, 3398, # 19_6_30
    564, 1585, 2606, 3627, # 19_19_2
    226, 1247, 2268, 3289, # 19_4_8
    361, 1382, 2403, 3424, # 19_7_40
    253, 1274, 2295, 3316, # 19_4_36
    412, 1433, 2454, 3475, # 19_9_26
    334, 1355, 2376, 3397, # 19_6_27
    89, 1110, 2131, 3152, # 19_1_35
    26, 1047, 2068, 3089, # 19_0_28
    279, 1300, 2321, 3342, # 19_5_12
    318, 1339, 2360, 3381, # 19_6_7
    341, 1362, 2383, 3404, # 19_6_38
    350, 1371, 2392, 3413, # 19_7_21
    31, 1052, 2073, 3094, # 19_0_34
    477, 1498, 2519, 3540, # 19_12_12
    501, 1522, 2543, 3564, # 19_13_14
    34, 1055, 2076, 3097, # 19_0_38
    80, 1101, 2122, 3143, # 19_1_25
    120, 1141, 2162, 3183, # 19_2_6
    570, 1591, 2612, 3633, # 19_19_15
    137, 1158, 2179, 3200, # 19_2_25
    115, 1136, 2157, 3178, # 19_1_62
    343, 1364, 2385, 3406, # 19_6_40
    153, 1174, 2195, 3216, # 19_2_43
    166, 1187, 2208, 3229, # 19_2_62
    402, 1423, 2444, 3465, # 19_9_11
    329, 1350, 2371, 3392, # 19_6_21
    122, 1143, 2164, 3185, # 19_2_8
    263, 1284, 2305, 3326, # 19_4_50
]

def main(dataset_dir, weights_dir, start_weight_num, end_weight_num, save_result='not_save'):
    epoch_list = range(int(start_weight_num), int(end_weight_num) + 1)
    run(CocoConfigImg1(), dataset_dir, weights_dir, test_img_list, epoch_list, save_result) 
    run_new(CocoConfigImg4(), dataset_dir, weights_dir, test_img_list, epoch_list, save_result)

if __name__ == "__main__":
    from sys import argv

    if len(argv) != 5 and len(argv) != 6:
        raise Exception("\nUsage:\n" + f"\t{argv[0]} [dataset-dir] [weights-dir] [start-weight-num] [end-weight-num] [save-result]")
    if len(argv) != 5 :
        main(argv[1], argv[2], argv[3], argv[4])
    main(argv[1], argv[2], argv[3], argv[4], argv[5])
