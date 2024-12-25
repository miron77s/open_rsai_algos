import os
import sys

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

from scipy import ndimage

import logging
logging.getLogger().setLevel(logging.ERROR)

# Root directory of the project
# ROOT_DIR = os.path.abspath("G:/mrcnn/")

# # Import Mask RCNN
# sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import model as modellib, utils

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
    NAME = "coco"

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

def main(output_path, weights_file, work_region_shp, path_to_raster):
    ds = gdal.Open(path_to_raster)
    print(ds.GetGeoTransform())
    print(ds.GetProjection())
    geo_transform = ds.GetGeoTransform()
    raster_x = geo_transform[0]
    raster_y = geo_transform[3]
    pixelSizeX = geo_transform[1]
    pixelSizeY = geo_transform[5]
    print('pixelSizeX', pixelSizeX)
    print('pixelSizeY', pixelSizeY)

    sr = osr.SpatialReference()
    sr.ImportFromWkt(ds.GetProjection())
    sr.ExportToProj4()

    sr = osr.SpatialReference()
    sr.ImportFromEPSG(3395)
    sr.ExportToWkt()

    print('RasterXSize', ds.RasterXSize)
    print('RasterYSize', ds.RasterYSize)
    print('RasterCount', ds.RasterCount)

    band_1 = ds.GetRasterBand(1)
    band_2 = ds.GetRasterBand(2)
    band_3 = ds.GetRasterBand(3)
    print('BlockSize', band_1.GetBlockSize())

    # Set up the shapefile driver 
    driver = ogr.GetDriverByName("ESRI Shapefile")
    # Open the data source in update mode if it exists, otherwise create a new data source

    if os.path.exists(os.path.join(output_path, "green.shp")):
        data_source = driver.Open(output_path, 1) # 1 is for update mode
        # Assuming data_source is already open in update mode
        layer_count = data_source.GetLayerCount()
        green_index = -1
        for i in range(layer_count):
            layer = data_source.GetLayerByIndex(i)
            if layer.GetName() == "green":
                green_index = i
                break

        if green_index != -1:
            data_source.DeleteLayer(green_index)
    else:
        # Create a new data source since it doesn't exist
        data_source = driver.CreateDataSource(output_path)

    # create the spatial reference system, WGS84
    srs =  osr.SpatialReference()
    srs.ImportFromEPSG(3395)
    # create one layer 
    layer = data_source.CreateLayer("green", srs, ogr.wkbPolygon)
    # Add an ID field
    idField = ogr.FieldDefn("id", ogr.OFTInteger)
    layer.CreateField(idField)
    # Create the feature and set values
    featureDefn = layer.GetLayerDefn()
    feature = ogr.Feature(featureDefn)
    # Create polygon
    poly = ogr.Geometry(ogr.wkbPolygon)

    net_h = 1024
    net_w = 1024

    model = mrcnn.model.MaskRCNN(
        mode="inference", 
        config=CocoConfigImg1(), 
        model_dir=os.getcwd()
    )

    print('weights_file:', weights_file)
    model.load_weights(
        filepath=weights_file, 
        by_name=True
    )

    tile_sizes = [1024, 2048, 4096]

    kernel = np.asarray(
        [[False, True, False],
         [True, True, True],
         [False, True, False]]
    )

    work_region_x1 = raster_x
    work_region_y1 = raster_y
    work_region_x2 = raster_x + ds.RasterXSize * pixelSizeX
    work_region_y2 = raster_y + ds.RasterYSize * pixelSizeY
    if work_region_shp != 'all' :
        print('set work region')
        work_region_file = ogr.Open(work_region_shp)
        work_region_shape = work_region_file.GetLayer(0)

        # Get the first feature of the shapefile
        work_region_feature = work_region_shape.GetFeature(0)

        # Use GetEnvelope to get the bounding box of the feature
        envelope = work_region_feature.GetGeometryRef().GetEnvelope()

        # Envelope returns a tuple (minX, maxX, minY, maxY)
        minX, maxX, minY, maxY = envelope

        # Top-left coordinate is (minX, maxY)
        work_region_x1 = minX
        work_region_y1 = maxY

        # Bottom-right coordinate is (maxX, minY)
        work_region_x2 = maxX
        work_region_y2 = minY

    print('work_region_x1', work_region_x1)
    print('work_region_y1', work_region_y1)
    print('work_region_x2', work_region_x2)
    print('work_region_y2', work_region_y2)

    work_region_w = work_region_x2 - work_region_x1
    work_region_h = work_region_y2 - work_region_y1
    print('work_region_w', work_region_w)
    print('work_region_h', work_region_h)
    mask_w = int(work_region_w / pixelSizeX)
    mask_h = int(work_region_h / pixelSizeY)
    print('mask_w', mask_w)
    print('mask_h', mask_h)
    mask = np.empty((mask_h, mask_w), dtype=bool)
    for tile_size in tile_sizes :
        print('raster to tile size', tile_size)
        w_range = range(int((mask_w - tile_size)  / (tile_size / 3)))
        h_range = range(int((mask_h - tile_size) / (tile_size / 3)))
        # w_range = range(3)
        # h_range = range(3)
        image_num = int((mask_w - tile_size)  / (tile_size / 3)) * int((mask_h - tile_size) / (tile_size / 3))
        start_x = (work_region_x1 - raster_x)
        start_y = (work_region_y1 - raster_y) * -1
        start_x = start_x / pixelSizeX
        start_y = start_y / pixelSizeY * -1
        printProgressBar(0, image_num, prefix = 'Progress:', suffix = '', length = 50)
        image_number = 0
        for block_j in h_range:
            block_y = int(block_j * (tile_size / 3))
            _block_y = int(block_y + tile_size)
            for block_i in w_range:
                image_number = image_number + 1
                printProgressBar(image_number, image_num, prefix = 'Progress:', suffix = '', length = 50)
                block_x = int(block_i * (tile_size / 3))
                _block_x = int(block_x + tile_size)

                block_1 = band_1.ReadAsArray(
                    xoff=start_x + block_x, 
                    yoff=start_y + block_y,
                    win_xsize=tile_size, 
                    win_ysize=tile_size
                )
                block_2 = band_2.ReadAsArray(
                    xoff=start_x + block_x, 
                    yoff=start_y + block_y,
                    win_xsize=tile_size, 
                    win_ysize=tile_size
                )
                block_3 = band_3.ReadAsArray(
                    xoff=start_x + block_x, 
                    yoff=start_y + block_y,
                    win_xsize=tile_size, 
                    win_ysize=tile_size
                )
                tile = np.empty((tile_size, tile_size, 3), dtype=np.uint8)
                tile[:, :, 0] = block_1
                tile[:, :, 1] = block_2
                tile[:, :, 2] = block_3
                stacked_img = np.stack((block_1, block_2, block_3), axis=-1)

                img = Image.fromarray(stacked_img)
                img = img.resize((net_h, net_w), Image.Resampling.LANCZOS)

                ret = model.detect([np.array(img)], verbose=0)
                # Get the results for the first image.
                r = ret[0]

                binary_mask = np.zeros((tile_size, tile_size), dtype=bool)
                for i in range(0, r['masks'].shape[2]) : 
                    x1, y1, x2, y2 =  r['rois'][i]
                    w = x2 - x1
                    h = y2 - y1
                    s = w * h
                    iterations = 1
                    if s < (tile_size / 16) ** 2 :
                        iterations = 0
                    elif s < (tile_size / 8) ** 2 :
                        iterations = 2
                    elif s < (tile_size / 4) ** 2 :
                        iterations = 7
                    elif s < (tile_size / 2) ** 2 :
                        iterations = 20
                    else :
                        iterations = 42
                    if iterations > 0 :
                        r['masks'][:, :, i] = ndimage.binary_dilation(r['masks'][:, :, i], kernel, iterations)
                    img = Image.fromarray(r['masks'][:, :, i])
                    img = img.resize((tile_size, tile_size), Image.Resampling.LANCZOS)
                    binary_mask += np.array(img)
                mask[block_y:_block_y, block_x:_block_x] += binary_mask

    print('find contours')
    mask = mask * 255
    numpy_array = mask.astype(np.uint8)
    mat_image = np.asarray(numpy_array)
    contours, _ = cv2.findContours(mat_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    raster_x += start_x * pixelSizeX
    raster_y += start_y * pixelSizeY
    print('make shape')
    contours_num = len(contours)
    printProgressBar(0, contours_num, prefix = 'Progress:', suffix = '', length = 50)
    contour_num = 0
    for contour in contours :
        contour_num = contour_num + 1
        printProgressBar(contour_num, contours_num, prefix = 'Progress:', suffix = '', length = 50)
        polygon = ogr.Geometry(ogr.wkbLinearRing)
        x0 = int(raster_x + contour[0][0][0] * pixelSizeX)
        y0 = int(raster_y + contour[0][0][1] * pixelSizeY)
        polygon.AddPoint(x0, y0)
        for i in range(1, contour.shape[0]) : 
            x = int(raster_x + contour[i][0][0] * pixelSizeX)
            y = int(raster_y + contour[i][0][1] * pixelSizeY)
            polygon.AddPoint(x, y)
        polygon.AddPoint(x0, y0)
        poly.AddGeometry(polygon)

    feature.SetGeometry(poly)
    feature.SetField("id", 1)
    layer.CreateFeature(feature)
    feature = None
    print('save shape')

    # Save and close DataSource
    data_source = None

if __name__ == "__main__":
    from sys import argv

    if len(argv) != 5:
        raise Exception("\nUsage:\n" + f"\t{argv[0]} [output-path] [weights-file] [work-region-shp] [path-to-raster]")
    main(argv[1], argv[2], argv[3], argv[4])
