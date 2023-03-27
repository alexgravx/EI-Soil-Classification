## importations ##

from sklearn.metrics import precision_score
import itertools
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import os
from scipy.ndimage import gaussian_filter
from skimage import io
from sklearn.cluster import KMeans
from osgeo import gdal
import numpy as np

## Creation des sets de tests et de train ##

src = '10089.tif'
path = './test/images/'
L_fin = []
L_fin2 = []
Tableau = [[0 for i in range(6)] for j in range(6)]
files_test = [f for f in os.listdir(
    path) if os.path.isfile(os.path.join(path, f))]
for i in range(len(files_test)):
    files_test[i] = os.path.join(path, files_test[i])
path = './test/masks/'
files_best = [f for f in os.listdir(
    path) if os.path.isfile(os.path.join(path, f))]
for i in range(len(files_best)):
    files_best[i] = os.path.join(path, files_best[i])
    
## Affichage du masque ##

def affichage_mask(path):
    with TiffFile(path) as tif:
        landcover = tif.asarray()

    cmap = colors.ListedColormap(
        ['k', 'magenta', 'red', 'khaki', 'green', 'darkgreen', 'lightgreen', 'grey', 'white', 'blue'])

    bounds = np.arange(10)
    norm = colors.BoundaryNorm(bounds, cmap.N)

    return (landcover, cmap, norm)

## Méthode des K-moyens ##

for k in range(len(files_test)):
    Clussters = []
    img_ds = gdal.Open(files_test[k], gdal.GA_ReadOnly)
    img = np.zeros((img_ds.RasterYSize, img_ds.RasterXSize, img_ds.RasterCount),
                   gdal_array.GDALTypeCodeToNumericTypeCode(img_ds.GetRasterBand(1).DataType))

    for b in range(img.shape[2]):
        img[:, :, b] = img_ds.GetRasterBand(b + 1).ReadAsArray()

    # Filtre gaussien
    sigma = 1
    img = gaussian_filter(img, sigma)

    new_shape = (img.shape[0] * img.shape[1], img.shape[2])
    X = img[:, :, :4].reshape(new_shape)

    k_means = cluster.KMeans(n_clusters=6)
    k_means.fit(X)

    X_cluster = k_means.labels_
    X_cluster = X_cluster.reshape(img[:, :, 0].shape)
    Clussters = X_cluster

    # Comptage de la fréquence d'apparition de chaque cluster
    total = [0, 0, 0, 0, 0, 0]
    for i in range(256):
        for j in range(256):
            total[Clussters[i][j]] += 1
    trr = sorted(total)

    Final = []
    for i in range(256):
        Fin = []
        for j in range(256):
            if trr.index(total[Clussters[i][j]]) == 0:
                Fin.append(0)  # no data
            if trr.index(total[Clussters[i][j]]) == 1:
                Fin.append(2)  # urrbain
            if trr.index(total[Clussters[i][j]]) == 2:
                Fin.append(9)  # eau
            if trr.index(total[Clussters[i][j]]) == 3:
                Fin.append(6)
            if trr.index(total[Clussters[i][j]]) == 4:
                Fin.append(3)
            if trr.index(total[Clussters[i][j]]) == 5:
                Fin.append(5)
        Final.append(Fin)
    landcover_mask, cmap_mask, norm_mask = affichage_mask(files_best[k])
    L_oh = []
    for i in range(256):
        L_abs = []
        for j in range(256):
            if landcover_mask[i][j] == 3:
                L_abs.append(3)
            if landcover_mask[i][j] == 6 or landcover_mask[i][j] == 7:
                L_abs.append(6)
            if landcover_mask[i][j] == 4 or landcover_mask[i][j] == 5:
                L_abs.append(5)
            if landcover_mask[i][j] == 2:
                L_abs.append(2)
            if landcover_mask[i][j] == 9:
                L_abs.append(9)
            if landcover_mask[i][j] == 0 or landcover_mask[i][j] == 1 or landcover_mask[i][j] == 8:
                L_abs.append(0)
        L_oh.append(L_abs)
    L_fin.append(Final)
    L_fin2.append(L_oh)
    
# Analyse des résultats avec une matrice de confusion

def plot_confusion_matrixx(y_true, y_pred):
    target_names = ['NADA', 'urbain', 'vegetation', 'forêts', 'champs', 'eau']
    cm = confusion_matrix(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f'Confusion matrix \n precision={precision:.3f}')
    plt.colorbar()
    tick_marks = np.arange(len(target_names))
    plt.xticks(tick_marks, target_names, rotation=45)
    plt.yticks(tick_marks, target_names)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j]), horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.show()


plt.figure(figsize=(10, 10))
L = []
F = []
for k in range(len(L_fin)):
    for i in range(256):
        for j in range(256):
            L.append(L_fin[k][i][j])
            F.append(L_fin2[k][i][j])
y_test = L
y_pred_brut = F
plot_confusion_matrixx(y_test, y_pred_brut)
