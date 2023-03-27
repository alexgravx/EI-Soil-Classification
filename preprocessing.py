## Algorithme 1, SVM ##

# Import datasets, classifiers and performance metrics
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split
from sklearn import datasets
import seaborn as sns
import pandas as pd
from affichage_data import *

# Recherche des images d'intérêt

def images_intérêt(datafiles, seuil):
  for file in datafiles:
    L = []
    mask = lecture_mask(file)
    a = (np.count_nonzero(mask == 2) > seuil)
    b = (np.count_nonzero(mask == 3) > seuil)
    c = (np.count_nonzero(mask == 4) > seuil)
    d = (np.count_nonzero(mask == 5) > seuil)
    e = (np.count_nonzero(mask == 6) > seuil)
    f = (np.count_nonzero(mask == 9) > seuil)
    if a*b*c*d*e*f == 1:
      L.append(file)
  if L == []:
    return False
  return L

# Import des données

datafiles = fichiers_train[:]
nb_fichiers = len(datafiles)
print(datafiles)

# Reshapes

def reshape_to_tableau(y_ligne):
  lignes = len(y_ligne)//256
  y_tableau = y_ligne.reshape(lignes, 256)
  return y_tableau

# Création de l'array ligne à partir d'une image

# Pour plusieurs images

X1 = np.concatenate((lecture_img(datafiles[0], 'train').reshape(65536,4),lecture_img(datafiles[1], 'train').reshape(65536,4)))
y1 = np.concatenate((lecture_mask(datafiles[0], 'train').reshape(65536,),lecture_mask(datafiles[1], 'train').reshape(65536,)))

# Pour une image

X2 = lecture_img(datafiles[0], 'train').reshape(65536,4)
y2 = lecture_mask(datafiles[0], 'train').reshape(65536,)

# Découpage des images (en parties séparées ou non)

X_data1 = np.concatenate((X1[:32768,:],X1[98304:,:]))
y_data1 = np.concatenate((y1[:32768], y1[98304:]))

X_data2 = X1[:25600,:]
y_data2 = y1[:25600]

print(np.shape(X_data1))
print(np.shape(y_data1))

# Mélange aléatoire des données d'entraînement

import random as rd

z = list(zip(X1,y1))
rd.shuffle(z)

X_alea, y_alea = zip(*z)
X_alea, y_alea = np.array(X_alea), np.array(y_alea)

# Decoupage

X_data_alea = X_alea[:5120,:]
y_data_alea = y_alea[:5120]

# Visualisation des données

df = pd.DataFrame()
data = pd.DataFrame(columns = ['no_data','clouds','urban','champs','arbres','forêt','vegetation','surface naturelle','neige','eau'])
df['label_zone'] = pd.DataFrame(data=y_data2)

hist = []
for i in range(10):
    a = len(df[df['label_zone'] == i])
    hist.append(a)

# Statistiques 

data.loc[0] = hist
data.head()