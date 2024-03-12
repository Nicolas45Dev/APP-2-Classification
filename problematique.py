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

    if False:
        images.generateRepresentation([])
        np.save("representation_coast.npy", images.representation_coast)
        np.save("representation_forest.npy", images.representation_forest)
        np.save("representation_street.npy", images.representation_street)


    if True:
        data_coast = np.load("representation_coast.npy")
        data_forest = np.load("representation_forest.npy")
        data_street = np.load("representation_street.npy")

        # visualisation des donnée en 3D
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(data_coast[:, 0], data_coast[:, 1], data_coast[:, 2], c='r', marker='o')
        ax.scatter(data_forest[:, 0], data_forest[:, 1], data_forest[:, 2], c='g', marker='o')
        ax.scatter(data_street[:, 0], data_street[:, 1], data_street[:, 2], c='b', marker='o')
        # Add legend
        ax.legend(['coast', 'forest', 'street'])
        # add axes labels
        ax.set_xlabel('Gray')
        ax.set_ylabel('Line no direction')
        ax.set_zlabel('Line V')
        # plt.show()

    all_representations = ClassificationData()

    # Bayes Classifier
    if False:
        # Bayes Classifier
        apriori = [0.3673, 0.3347, 0.298]
        cost = [[0, 1, 1], [1, 0, 1], [1, 1, 0]]
        bg1 = classifiers.BayesClassify_APP2(data2train=all_representations, data2test=all_representations,
                                             apriori=apriori, costs=cost,
                                             experiment_title='probabilités gaussiennes',
                                             gen_output=True, view=True)
    # PPV Classifier
    if True:
        data_coast = np.load("representation_coast.npy")
        data_forest = np.load("representation_forest.npy")
        data_street = np.load("representation_street.npy")

        start = 0
        stop = 200

        temp = []
        temp.append(data_coast[start:stop])
        temp.append(data_forest[start:stop])
        temp.append(data_street[start:stop])
        traning_data = ClassificationData(temp)

        start = 201
        stop = 292

        temp = []
        temp.append(data_coast[start:stop])
        temp.append(data_forest[start:stop])
        temp.append(data_street[start:stop])
        test_data = ClassificationData(temp)

        # PPV
        if True:
            nb_representants = 1

            # Tests
            if False:
                # 2n = 5.47945205479452 %
                # 3n = 5.821917808219178 %
                # 4n = 6.392694063926941 %

                # 8n = 7.6923076923076925 %

                for nb_neighbors in range(1, 30):
                    classifiers.PPVClassify_APP2(data2train=traning_data, data2test=test_data, n_neighbors=nb_neighbors,
                                                 experiment_title='PPV + (' + str(nb_representants) + ')representants + (' + str(nb_neighbors) + ')neighbors',
                                                 useKmean=False, n_representants=nb_representants,
                                                 gen_output=True, view=True)
            # Run
            if True:
                nb_neighbors = 8
                classifiers.PPVClassify_APP2(data2train=traning_data, data2test=test_data, n_neighbors=nb_neighbors,
                                             experiment_title='PPV + (' + str(nb_representants) + ')representants + (' + str(nb_neighbors) + ')neighbors',
                                             useKmean=False, n_representants=nb_representants,
                                             gen_output=True, view=True)

        # PPV with kmean
        if False:
            # 9r 1n = 9%
            # 12r 1n = 8.9%

            nb_neighbors = 1

            # Test
            if False:
                for nb_representants in range(1, 30):
                    classifiers.PPVClassify_APP2(data2train=traning_data, data2test=test_data, n_neighbors=nb_neighbors,
                                                 experiment_title='PPV + KMEAN + (' + str(nb_representants) + ')representants + (' + str(nb_neighbors) + ')neighbors',
                                                 useKmean=True, n_representants=nb_representants,
                                                 gen_output=True, view=True)

            # Run
            if True:
                nb_representants = 10
                classifiers.PPVClassify_APP2(data2train=traning_data, data2test=test_data, n_neighbors=nb_neighbors,
                                             experiment_title='PPV + KMEAN + (' + str(nb_representants) + ')representants + (' + str(nb_neighbors) + ')neighbors',
                                             useKmean=True, n_representants=nb_representants,
                                             gen_output=True, view=True)

    # ML Classification
    if True:
        # Exemple de RN
        n_neurons = 6
        n_layers = 2
        # Classification NN
        # get 20% of the data for testing in random
        train_data, test_data, train_labels, test_labels = ttsplit(all_representations.data1array,
                                                                   all_representations.labels1array, test_size=.99,
                                                                   random_state=1)
        data_test = {'data': test_data, 'label': test_labels}
        nn1 = classifiers.NNClassify_APP2(data2train=all_representations, data2test=data_test,
                                          n_layers=n_layers, n_neurons=n_neurons, innerActivation='sigmoid',
                                          outputActivation='softmax', optimizer=Adam(learning_rate=0.15),
                                          loss='mae',
                                          metrics=['accuracy'],
                                          callback_list=[],
                                          experiment_title='NN Simple',
                                          n_epochs=1000, savename='problematic_APP2',
                                          ndonnees_random=1000, train=0.7, gen_output=True, view=True)

    if True:
        plt.show()

######################################
if __name__ == '__main__':
    problematique_APP2()
