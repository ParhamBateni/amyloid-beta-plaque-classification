import h5py
import os
from time import time
from PIL import Image
print(os.getcwd())
start_time = time()
# with h5py.File("pbt-plaque-analysis/data/images/2012-021_F2_AB_Image.hdf5", "r") as f:
#     end_time = time()
#     print(f"Time taken to open file: {end_time - start_time} seconds")
#     print(f['plaques']['0'].attrs['roundness'])
#     print(f['plaques']['0'].attrs['area'])

image_class = os.listdir("data/labeled_images")[0]
image_name = os.listdir(os.path.join("data/labeled_images", image_class))[0]
file_path = os.path.join("data/labeled_images", image_class, image_name)
print(Image.open(file_path).size)