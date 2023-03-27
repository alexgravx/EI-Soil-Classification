# Résultats de performance

# Précisions et recall
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred_brut))

# Matrice de confusion
from sklearn.metrics import confusion_matrix
import itertools
from sklearn.metrics import precision_score

def plot_confusion_matrix(y_true, y_pred):
    target_names = ['urban', 'champs', 'forêts','herbes rases', 'eau']
    cm = confusion_matrix(y_true, y_pred)
    import matplotlib.pyplot as plt
    import numpy as np
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion matrix')
    plt.colorbar()
    #tick_marks = np.arange(len(set(y_true)))
    tick_marks = np.arange(len(target_names))
    plt.xticks(tick_marks, target_names, rotation=45)
    plt.yticks(tick_marks, target_names)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]),range(cm.shape[1])):
      plt.text(j, i, format(cm[i, j]), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")
    plt.show()

plt.figure(figsize=(10, 10))
plot_confusion_matrix(regroup(y_test), y_pred_brut)