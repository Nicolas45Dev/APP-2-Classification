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

        all_representations = ClassificationData()

    # Bayes Classifier
    if False:
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
    if True:
        # Exemple de RN
        n_neurons = 6
        n_layers = 2
        # Classification NN
        #get 20% of the data for testing in random
        train_data, test_data, train_labels, test_labels = ttsplit(all_representations.data1array, all_representations.labels1array, test_size=0.2, random_state=1)
        data_test = {'data': test_data, 'label': test_labels}
        nn1 = classifiers.NNClassify_APP2(data2train=all_representations, data2test=data_test,
                                          n_layers=n_layers, n_neurons=n_neurons, innerActivation='sigmoid',
                                          outputActivation='softmax', optimizer=Adam(learning_rate=0.25),
                                          loss='mse',
                                          metrics=['accuracy'],
                                          callback_list=[],
                                          experiment_title='NN Simple',
                                          n_epochs=1000, savename='problematic_APP2',
                                          ndonnees_random=1000, train=0.8, gen_output=True, view=True)
    plt.show()

######################################
if __name__ == '__main__':
    problematique_APP2()
