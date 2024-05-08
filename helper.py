import os
from skimage import io, color
import numpy as np

# Function to load images from directory
def load_images_from_folder(folder, is_gray = True):
    images = []
    for filename in os.listdir(folder):
        img = io.imread(os.path.join(folder,filename))
        if img is not None:
            if is_gray:
                images.append(color.rgb2gray(img / 255))
            else:    
                images.append(img / 255)
    return images



