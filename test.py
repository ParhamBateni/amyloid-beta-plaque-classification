import h5py
import os
from time import time
from PIL import Image
print(os.getcwd())
start_time = time()
with h5py.File("../jmoreirakanale/plaque-analysis/result/images/2012-099_Occipital_AB_Image.hdf5", "r") as f:
    print(f['plaques']['25171'].keys())
    picture = f['plaques']['25171']['plaque'][:]
    print(picture.shape)
with h5py.File("../jmoreirakanale/plaque-analysis/result/labeled_images/2012-099_Occipital_AB_Image.hdf5", "r") as f:
    print(f['plaques']['25171'].keys())
    picture = f['plaques']['25171']['plaque'][:]
    print(picture.shape)
