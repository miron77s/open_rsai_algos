# Open Remote Sensing AI Library Algorithms (OpenRSAI-Algos)

## Description

The OpenRSAI-Algos provide greenery and hydro detection on satelite imagery.

## Requirements

All instructions below are tailored for Ubuntu 22.04 users. If you are using a different operating system, please adjust the commands accordingly.

Building detection and reconstruction algorithms are implemented in [Open Remote Sensing AI Library Core](https://github.com/miron77s/open_rsai). To install all instrument pipelines follow the  [requirements](https://github.com/miron77s/open_rsai#requirements) and pull all it's submodules:


```
git clone https://github.com/miron77s/open_rsai_algos
cd open_rsai_algos
git submodule update --init --recursive
```

However if the standalone hydro+greenery detector is required simply clone this repository and follow the installation guide.

### Hardware

OpenRSAI-Algos utilities require the following hardware:

 - 16 Gb RAM.
 - NVIDIA GeForce 4070 (minimum 12Gb GPU memory).

 For both training and inference it is recommended to use NVIDIA GeForce 4090.

## Installation

Follow these steps to set up the environment and install OpenRSAI-Algos on your machine.

### Prerequisites

- Make sure you have [Anaconda](https://www.anaconda.com/products/individual) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html) installed on your system to manage your environments and packages.

### Setting up the environment

1. Open your terminal or Anaconda Prompt.
2. Create a new Conda environment with the necessary dependencies by running the following command:

```
conda create -n open_rsai_detectors python=3.9.0 gdal=3.6.2 numpy==1.19.5
```

3. Activate the newly created environment:

```
conda activate open_rsai_detectors
```

### Installing CUDA and TensorFlow and other dependencies

1. Within the active environment, install CUDA 11.0 and TensorFlow 2.5.0:

```
pip install tensorflow==2.5.0 --ignore-installed
conda install -c conda-forge cudatoolkit=11.0
```

The `--ignore-installed` flag ensures that the specified version of TensorFlow is installed even if an existing version is present.

2. Install additional dependencies:

```
pip install scipy==1.6.0 scikit-image==0.18.0 ipython==7.34.0 opencv-python pycocotools numpy==1.19.5
```

3. Install `h5py`, which is a requirement for handling HDF5 weights files:

```
conda install h5py
```

### Installing a customized Mask-RCNN

Navigate to the `mrcnn_library` repository's directory and run `setup.py` script:

```
cd mrcnn_library
python setup.py install
```

After completing these steps, OpenRSAI-Algos should be installed and ready to use in your `open_rsai_detectors` Conda environment.

## Training and testing hydrography and greenery detectors

### Train hydro
Usage:
```
    train_hydro.py [dataset-dir] [init-weights]
```

 * [dataset-dir] - path to directory with markup and tiles
 * [init-weights] - starting weight for training

Example of use:
```
    python train_hydro.py ./mrcnn/data/hydro/ ./weights/mask_rcnn_coco.h5
```

#### Note
The training result will be saved in the directory from which the script was launched
in a folder with a name in the format 'coco[current-date-time]'

### Train green
Usage:
```
    train_green.py [dataset-dir] [init-weights]
```

 * [dataset-dir] - path to directory with markup and tiles
 * [init-weights] - starting weight for training

Example of use:
```
    python train_green.py ./mrcnn/data/green/ ./weights/mask_rcnn_coco.h5
```

#### Note
The training result will be saved in the directory from which the script was launched
in a folder with a name in the format 'coco[current-date-time]'

### Test hydro
Testing: collecting statistics and plotting graphs, as well as saving segmentation results and
debugging information

Usage:
```
    test_hydro.py [dataset-dir] [weights-dir] [start-weight-num] [end-weight-num] [save-result]
```

 * [dataset-dir] - path to directory with markup and tiles
 * [weights-dir] - path to the directory with the tested weights
 * [start-weight-num] - weight number from which testing will begin
 * [end-weight-num] - weight number at which the test will be completed
 * [save-result] - optional parameter enabling/disabling saving of segmentation results

Example of use:
```
    python test_hydro.py data/hydro test_hydro 1 150 not_save
```

#### Note
Test results are saved in the directory '[weights-dir]/log'
The segmentation results are saved to the directory '[weights-dir]/result/[weight-name]'
To save the segmentation results, you must specify an optional parameter [save-result]
in the form 'save' or 'not_save'. Saving results slows down testing significantly.
By default, saving segmentation results is disabled. It is recommended to run the script
in 'spyder' for easy viewing of test results

### Test green
Testing: collecting statistics and plotting graphs, as well as saving segmentation results and
debugging information

Usage:
    test_green.py [dataset-dir] [weights-dir] [start-weight-num] [end-weight-num] [save-result]

 * [dataset-dir] - path to directory with markup and tiles
 * [weights-dir] - path to the directory with the tested weights
 * [start-weight-num] - weight number from which testing will begin
 * [end-weight-num] - weight number at which the test will be completed
 * [save-result] - optional parameter enabling/disabling saving of segmentation results

Example of use:
```
    python test_green.py data/green test_green 1 150 not_save
```

#### Note
Test results are saved in the directory '[weights-dir]/log'
The segmentation results are saved to the directory '[weights-dir]/result/[weight-name]'
To save the segmentation results, you must specify an optional parameter [save-result]
in the form 'save' or 'not_save'. Saving results slows down testing significantly.
By default, saving segmentation results is disabled. It is recommended to run the script
in 'spyder' for easy viewing of test results

### Detect hydro
Segmenting the water layer on a raster and saving the resulting layer in shapefile format
with georeferencing

Usage:
```
    detect_hydro.py [output-path] [weights-file] [work-region-shp] [path-to-raster]
```

 * [output-path] - path to shape with results
 * [weights-file] - path to weights
 * [work-region-shp] - path to shape defining area of ​​interest
 * [path-to-raster] - path to the raster on which detection will be performed

Examples of use:
```
    python detect_hydro.py hydro.shp ./weights/mask_rcnn_coco_0115.h5 work_region_025.shp Kursk_Bing_19.tif
    python detect_hydro.py hydro.shp ./weights/mask_rcnn_coco_0115.h5 all Kursk_Bing_19.tif
```

### Detect green
Segmenting the vegetation layer on a raster and saving the resulting layer in shapefile format
with georeferencing

Usage:
```
    detect_green.py [output-path] [weights-file] [work-region-shp] [path-to-raster]
```

 * [output-path] - path to shape with results
 * [weights-file] - path to weights
 * [work-region-shp] - path to shape defining area of ​​interest
 * [path-to-raster] - path to the raster on which detection will be performed

Examples of use:

```
    python detect_green.py green.shp ./weights/mask_rcnn_coco_0073.h5 work_region_025.shp Kursk_Bing_19.tif
    python detect_green.py green.shp ./weights/mask_rcnn_coco_0073.h5 all Kursk_Bing_19.tif
```

## Special Thanks

We wish to thank Innovations Assistance Fund (Фонд содействия инновациям, https://fasie.ru/)
for their support in our project within Code-AI program (https://fasie.ru/press/fund/kod-ai/).