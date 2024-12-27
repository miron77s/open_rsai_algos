import mrcnn.open_rsai_detect
from mrcnn.open_rsai_config import HydroDetectConfig

if __name__ == "__main__":
    from sys import argv

    if len(argv) != 5:
        raise Exception("\nUsage:\n" + f"\t{argv[0]} [output-path] [weights-file] [work-region-shp] [path-to-raster]")

    config = HydroDetectConfig()
    mrcnn.open_rsai_detect.hydro(argv[1], argv[2], argv[3], argv[4], config)
