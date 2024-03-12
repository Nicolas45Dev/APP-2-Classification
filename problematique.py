"""
Script de départ de la problématique
Problématique APP2 Module IA S8
"""

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

from helpers.ImageCollection import ImageCollection
from helpers import classifiers
from helpers.ClassificationData import ClassificationData
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split as ttsplit
from helpers.ClassificationData import ClassificationData

#######################################
def problematique_APP2():
    images = ImageCollection(load_all=True)
    dataToTreat = []
    # Génère une liste de N images, les visualise et affiche leur histo de couleur
    if True:
        # images = ImageCollection(load_all=True)
        #images.generateRepresentation([])

        #np.save("representation_coast.npy", images.representation_coast)
        #np.save("representation_forest.npy", images.representation_forest)
        #np.save("representation_street.npy", images.representation_street)


        all_representations = ClassificationData()

    # Bayes Classifier
    if True:
        # Bayes Classifier
        apriori = [1 / 3, 1 / 3, 1 / 3]
        cost = [[0, 1, 1], [1, 0, 1], [1, 1, 0]]
        bg1 = classifiers.BayesClassify_APP2(data2train=all_representations, data2test=all_representations,
                                             apriori=apriori, costs=cost,
                                             experiment_title='probabilités gaussiennes',
                                             gen_output=True, view=True)
    # PPV Classifier
    if False:
        ppv1 = classifiers.PPVClassify_APP2(data2train=all_representations, n_neighbors=1,
                                            experiment_title='1-PPV avec données orig comme représentants',
                                            gen_output=True, view=True)
        # 1-mean sur chacune des classes
        # suivi d'un 1-PPV avec ces nouveaux représentants de classes
        ppv1km1 = classifiers.PPVClassify_APP2(data2train=all_representations, data2test=all_representations, n_neighbors=1,
                                               experiment_title='1-PPV sur le 1-moy',
                                               useKmean=True, n_representants=7,
                                               gen_output=True, view=True)
    # ML Classification
    if False:
        # Exemple de RN
        n_neurons = 6
        n_layers = 2
        # Classification NN
        nn1 = classifiers.NNClassify_APP2(data2train=all_representations, data2test=all_representations,
                                          n_layers=n_layers, n_neurons=n_neurons, innerActivation='sigmoid',
                                          outputActivation='softmax', optimizer=Adam(learning_rate=0.2),
                                          loss='binary_crossentropy',
                                          metrics=['accuracy'],
                                          callback_list=[],
                                          experiment_title='NN Simple',
                                          n_epochs=1000, savename='problematic_APP2',
                                          ndonnees_random=5000, gen_output=True, view=True)

    #plt.show()

######################################
if __name__ == '__main__':
    problematique_APP2()
