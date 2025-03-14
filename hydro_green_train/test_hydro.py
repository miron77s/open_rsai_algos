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

CLASS_NAMES = ['BG', 'hydro']

class CocoConfigImg1(Config):
    """Configuration for training on MS COCO.
    Derives from the base Config class and overrides values specific
    to the COCO dataset.
    """
    # Give the configuration a recognizable name
    NAME = "hydro"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 1

    # Uncomment to train on 8 GPUs (default is 1)
    GPU_COUNT = 1

    # Number of classes (including background)
    NUM_CLASSES = len(CLASS_NAMES)
    
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

class CocoConfigImg3(CocoConfigImg1):
    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 3
    
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
    # image_ids = dataset.load_coco(dataset_dir, val_type, year="2014")
    # dataset.prepare()
 
    weights_dir = target_dir + "mask_rcnn_hydro_*epoch*.h5"

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
                    iterations = 1
                elif s < 256 * 256 :
                    iterations = 5
                elif s < 512 * 512 :
                    iterations = 25
                else :
                    iterations = 40
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
                            'water: ' + str(int(round(max_iou, 2) * 100)) + '% '
                            ' (left_x: ' + str(_x1)+ ' '*(5-len(str(_x1))) + 'top_y: ' + str(_y1) + ' '*(5-len(str(_y1))) +
                            'width:' + str(_w) + ' '*(5-len(str(_w))) + 'height:' + str(_h) + ' '*(5-len(str(_h))) + ')\n')
            
                cur_iou_list.append(max_iou)  #* best_s
                cur_s_list.append(best_s)
 
            iou = float('nan')
            if len(cur_iou_list) > 0 : #and s_sum > 0 :
                iou = sum(cur_iou_list) / len(cur_iou_list)
            
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
    # image_ids = dataset.load_coco(dataset_dir, val_type, year="2014")
    # dataset.prepare()
   
    weights_dir = target_dir + "mask_rcnn_hydro_*epoch*.h5"

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
            
        image_num = len(test_img_list)
        print('image_num', image_num)
        printProgressBar(0, image_num, prefix = 'Progress:', suffix = '', length = 50)
        for image_number in range(0, image_num, 3) :
            printProgressBar(image_number + 3, image_num, prefix = 'Progress:', suffix = '', length = 50)
            image_id = test_img_list[image_number]
            img_path, image, gt_class_ids, gt_bbox, gt_mask = dataset.load_image_gt(test_img_list[image_number])
            _, image1, _, _, _ = dataset.load_image_gt(test_img_list[image_number + 1])
            _, image2, _, _, _ = dataset.load_image_gt(test_img_list[image_number + 2])
            if len(gt_bbox) == 0 :
                continue
            gt_data = {}
            gt_data['masks'] = gt_mask
            gt_data['rois'] = gt_bbox
            gt_data['class_ids'] = gt_class_ids
            
            ret = model.detect([image, image1, image2], verbose=0)

            save_path = target_path + Path(img_path).stem + ".png"
        
            kernel = np.asarray(
                [[False, True, False],
                [True, True, True],
                [False, True, False]]
            )
            
            for n in range(0, 3) :
                for i in range(0, ret[n]['masks'].shape[2]) : 
                    x1, y1, x2, y2 =  ret[n]['rois'][i]
                    w = x2 - x1
                    h = y2 - y1
                    s = w * h
                    iterations = 1
                    if s < 64 * 64 :
                        iterations = 0
                    elif s < 128 * 128 :
                        iterations = 1
                    elif s < 256 * 256 :
                        iterations = 5
                    elif s < 512 * 512 :
                        iterations = 25
                    else :
                        iterations = 40
                    if iterations > 0 :
                        ret[n]['masks'][:, :, i] = ndimage.binary_dilation(ret[n]['masks'][:, :, i], kernel, iterations)
            
            sum_mask = zeros((1024, 1024), np.int)
            for n in range(0, 3) :
                for i in range(0, ret[n]['masks'].shape[2]) : 
                    sum_mask += ret[n]['masks'][:, :, i].astype(int)
            binary_mask = sum_mask > 2
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
                iou = sum(cur_iou_list) / len(cur_iou_list)
            
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
    1816, 4968, 8120, # 19_32_23
    2053, 5205, 8357, # 19_36_35
    1407, 4559, 7711, # 19_24_54
    2382, 5534, 8686, # 19_43_11
    2322, 5474, 8626, # 19_42_1
    2235, 5387, 8539, # 19_40_19
    2038, 5190, 8342, # 19_36_17
    2329, 5481, 8633, # 19_42_11
    53, 3205, 6357, # 19_0_58
    2133, 5285, 8437, # 19_38_12
    1199, 4351, 7503, # 19_21_8
    2331, 5483, 8635, # 19_42_15
    1733, 4885, 8037, # 19_30_53
    2000, 5152, 8304, # 19_35_37
    1997, 5149, 8301, # 19_35_34
    2233, 5385, 8537, # 19_40_16
    2891, 6043, 9195, # 19_52_19
    2139, 5291, 8443, # 19_38_20
    796, 3948, 7100, # 19_14_8
    2105, 5257, 8409, # 19_37_40
    1837, 4989, 8141, # 19_32_46
    1734, 4886, 8038, # 19_30_54
    2098, 5250, 8402, # 19_37_26
    1551, 4703, 7855, # 19_27_31
    2230, 5382, 8534, # 19_40_5
    1512, 4664, 7816, # 19_26_47
    2284, 5436, 8588, # 19_41_18
    2384, 5536, 8688, # 19_43_13
    1460, 4612, 7764, # 19_25_51
    2109, 5261, 8413, # 19_37_48
    2297, 5449, 8601, # 19_41_33
    2093, 5245, 8397, # 19_37_21
    1958, 5110, 8262, # 19_34_52
    1404, 4556, 7708, # 19_24_51
    2045, 5197, 8349, # 19_36_27
    485, 3637, 6789, # 19_8_41
    2846, 5998, 9150, # 19_51_31
    51, 3203, 6355, # 19_0_56
    1124, 4276, 7428, # 19_19_50
    56, 3208, 6360, # 19_0_61
    608, 3760, 6912, # 19_10_53
    2179, 5331, 8483, # 19_39_7
    2328, 5480, 8632, # 19_42_9
    606, 3758, 6910, # 19_10_51
    274, 3426, 6578, # 19_4_51
    499, 3651, 6803, # 19_8_56
    742, 3894, 7046, # 19_13_11
    2286, 5438, 8590, # 19_41_20
    2280, 5432, 8584, # 19_41_11
    2191, 5343, 8495, # 19_39_21
    1529, 4681, 7833, # 19_27_7
    2013, 5165, 8317, # 19_35_53
    2247, 5399, 8551, # 19_40_34
    2330, 5482, 8634, # 19_42_14
    2282, 5434, 8586, # 19_41_14
    2232, 5384, 8536, # 19_40_8
    1935, 5087, 8239, # 19_34_26
    1621, 4773, 7925, # 19_28_47
    680, 3832, 6984, # 19_12_10
    2279, 5431, 8583, # 19_41_10
    1785, 4937, 8089, # 19_31_49
    2153, 5305, 8457, # 19_38_39
    1961, 5113, 8265, # 19_34_57
    484, 3636, 6788, # 19_8_40
    2364, 5516, 8668, # 19_42_54
]
 
def main(dataset_dir, weights_dir, start_weight_num, end_weight_num, save_result='not_save'):
    epoch_list = range(int(start_weight_num), int(end_weight_num) + 1)
    run(CocoConfigImg1(), dataset_dir, weights_dir, test_img_list, epoch_list, save_result) 
    run_new(CocoConfigImg3(), dataset_dir, weights_dir, test_img_list, epoch_list, save_result)

if __name__ == "__main__":
    from sys import argv

    if len(argv) != 5 and len(argv) != 6:
        raise Exception("\nUsage:\n" + f"\t{argv[0]} [dataset-dir] [weights-dir] [start-weight-num] [end-weight-num] [save-result]")
    if len(argv) == 5 :
        main(argv[1], argv[2], argv[3], argv[4])
    main(argv[1], argv[2], argv[3], argv[4], argv[5])
