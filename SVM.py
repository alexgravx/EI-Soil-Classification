from preprocessing import *
from sklearn import datasets, svm, metrics

# Module sklearn gaussien
from sklearn.svm import SVC

# Création du classifier
classifier_rbf = SVC(kernel = 'rbf', random_state = 0)

# Entrainement du dataset
classifier_rbf.fit(X_data2, y_data2)

# Module sklearn sigmoide
from sklearn.svm import SVC

# Création du classifier
classifier_sig = SVC(kernel = 'sigmoid', random_state = 0)

# Entrainement du dataset
classifier_sig.fit(X_data1, y_data1)

# Module sklearn polynomial
from sklearn.svm import SVC

# Création du classifier
classifier_poly = SVC(kernel = 'poly', random_state = 0)

# Entrainement du dataset
classifier_poly.fit(X_data1, y_data1)

# Module sklearn linéaire
from sklearn.svm import SVC

# Création du classifier
classifier_linear = SVC(kernel = 'linear', random_state = 0)

# Entrainement du dataset
classifier_linear.fit(X_data1, y_data1)

# Lecture de l'image

datatest = fichiers_test[:]

print(datatest)
X_test = lecture_img(datatest[0], 'test').reshape(65536,4)
y_test = lecture_mask(datatest[0], 'test').reshape(65536,)

#Prediction sur le Test set. RBF et poly semblent être les plus efficaces

y_pred_brut = classifier_rbf.predict(X_test)

# Modification des classes (regroupement)

def regroup(array):
    array[array == 1] = 0
    array[array == 8] = 0
    array[array == 5] = 4
    array[array == 7] = 6
    return array

y_pred_brut = regroup(y_pred_brut)
y_test = regroup(y_test)

# Affichage 

fig, (ax1,ax2) = plt.subplots(1,2, figsize = (15,15))
#Masque voulu
plot.show(lecture_mask(datatest[0], 'test'), cmap = cmap_mask, norm = norm_mask, ax = ax1)
# Masque obtenu
plot.show(reshape_to_tableau(y_pred_brut), cmap = cmap_mask, norm = norm_mask, ax = ax2)