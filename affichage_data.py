import numpy as np
from matplotlib import colors
import matplotlib.pyplot as plt
from tifffile import TiffFile

import os
from os import listdir

import rasterio
from rasterio.plot import show
from rasterio import plot

# Liste des fichiers

path_train_masks = './dataset/train/masks/'
path_train_images = './dataset/train/images/'
path_test_masks = './dataset/test/masks/'
path_test_images = './dataset/test/images/'


fichiers_train = [f for f in listdir(path_train_images)]
fichiers_test = [f for f in listdir(path_test_images)]

# Affichage des données

# Fonctions lecture permettent de récupérer l'array [[R,G,B,NIR] for pixel in image]

def lecture_img(path, type):
    if type == 'train':
      with TiffFile(path_train_images+path) as tif:
          landcover = tif.asarray()
    elif type =='test':
      with TiffFile(path_test_images+path) as tif:
          landcover = tif.asarray()
    return landcover

def lecture_mask(path, type):
    if type == 'train':
      with TiffFile(path_train_masks+path) as tif:
          landcover = tif.asarray()
    elif type == 'test':
      with TiffFile(path_test_masks+path) as tif:
          landcover = tif.asarray()
    return landcover

# Fonction normalisation
def normalize(x, lower, upper):
    """Normalize an array to a given bound interval"""

    x_max = np.max(x)
    x_min = np.min(x)

    m = (upper - lower) / (x_max - x_min)
    x_norm = (m * (x - x_min)) + lower

    return x_norm

# Fonction affichage de l'image désirée

def affichage_img(path, type):
    if type == 'train':
      # Normalize each band separately
      dataset = rasterio.open(path_train_images+path).read([3,2,1])
    elif type == 'test':
      dataset = rasterio.open(path_test_images+path).read([3,2,1])
    data_norm = np.array([normalize(dataset[i,:,:], 0, 255) for i in range(dataset.shape[0])])
    data_rgb = data_norm.astype("uint8")
    return data_rgb

# Fonction affichage du masque désiré

def affichage_mask(path, type):
    if type == 'train':
      with TiffFile(path_train_masks+path) as tif:
          landcover = tif.asarray()
    elif type == 'test':
      with TiffFile(path_test_masks+path) as tif:
          landcover = tif.asarray()

    cmap = colors.ListedColormap(['k', 'magenta', 'red', 'khaki', 'green', 'darkgreen', 'lightgreen', 'grey', 'white', 'blue'])

    bounds = np.arange(10)
    norm = colors.BoundaryNorm(bounds, cmap.N)
    
    return (landcover, cmap, norm)

def train_or_test(src):
    number = int(src[:-4])
    if number < 10087:
      return 'train'
    elif number >= 10087:
      return 'test'
  
# Exemple d'affichage

src = '3.tif'

type = train_or_test(src)

landcover_mask, cmap_mask, norm_mask = affichage_mask(src,type)
data_rgb, cmap_img = affichage_img(src,type), colors.ListedColormap(['blue', 'green', 'red', 'k'])

fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize = (20,7))

#image
plot.show(data_rgb, cmap = cmap_img, ax = ax1)
#masque
plot.show(landcover_mask, cmap = cmap_mask, norm = norm_mask, ax = ax2)
#histogramme
plot.show_hist(data_rgb, bins=800, histtype='stepfilled',lw=0.0, stacked=False, alpha=0.9, ax=ax3)