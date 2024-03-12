"""
Script de départ de la problématique
Problématique APP2 Module IA S8
"""

import matplotlib.pyplot as plt
import numpy as np

from helpers.ImageCollection import ImageCollection
from helpers import classifiers
from helpers.ClassificationData import ClassificationData
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split as ttsplit
from helpers.ClassificationData import ClassificationData

#######################################
def showRepresentation():
    axisName = ['gris', 'rouge', 'vert', 'bleu', 'angle median', 'angle iqr']
    nb_dimension = 6

    coast = np.load("representation_coast.npy")[:275]
    forest = np.load("representation_forest.npy")[:275]
    street = np.load("representation_street.npy")[:275]

    for a in range(nb_dimension):
        for b in range(a + 1, nb_dimension):
            for c in range(b + 1, nb_dimension):

                fig = plt.figure()
                ax = fig.add_subplot(projection='3d')
                ax.set_title(axisName[a] + ' + ' + axisName[b] + ' + ' + axisName[c])
                ax.set_xlabel(axisName[a])
                ax.set_ylabel(axisName[b])
                ax.set_zlabel(axisName[c])


                # Coast
                ax.scatter(coast[:,a], coast[:,b], coast[:,c], c='b', label='coast')

                # Forest
                ax.scatter(forest[:, a], forest[:, b], forest[:, c], c='g', label='forest')

                # Street
                ax.scatter(street[:, a], street[:, b], street[:, c], c='r', label='street')

                ax.legend()
                ax.set_xlim([0, 1])
                ax.set_ylim([0, 1])
    plt.show()



















def problematique_APP2():
    images = ImageCollection(load_all=True)
    dataToTreat = []
    # Génère une liste de N images, les visualise et affiche leur histo de couleur
    if True:
        # Analyser quelques images pour développer des pistes pour le choix de la représentation
        # N = 6
        # im_list = images.get_samples(N)

        # images.generateRepresentation([])
        # np.save("representation_coast.npy", images.representation_coast)
        # np.save("representation_forest.npy", images.representation_forest)
        # np.save("representation_street.npy", images.representation_street)
        # representation_coast = np.load("representation_coast.npy")
        # representation_forest = np.load("representation_forest.npy")
        # representation_street = np.load("representation_street.npy")
        showRepresentation()
        all_representations = ClassificationData()
        # images.do_pca_coast(representation_coast)
        # images.do_pca_forest(representation_forest)
        # images.do_pca_street(representation_street)

    # Bayes Classifier
    if False:
        # Bayes Classifier
        bayes1 = classifiers.BayesClassify_APP2(data2train=all_representations, data2test=all_representations,
                                                experiment_title='Bayes',
                                                gen_output=True, view=True)
    # PPV Classifier
    if True:
        # 1-mean sur chacune des classes
        # suivi d'un 1-PPV avec ces nouveaux représentants de classes
        ppv1km1 = classifiers.PPVClassify_APP2(data2train=all_representations, data2test=all_representations, n_neighbors=1,
                                               experiment_title='1-PPV sur le 1-moy',
                                               useKmean=True, n_representants=7,
                                               gen_output=True, view=False)
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
                                          callback_list=[],  # TODO à compléter L2.E4
                                          experiment_title='NN Simple',
                                          n_epochs=1000, savename='problematic_APP2',
                                          ndonnees_random=5000, gen_output=True, view=True)

######################################
if __name__ == '__main__':
    problematique_APP2()
    plt.show()
