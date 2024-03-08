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

#######################################
def problematique_APP2():
    images = ImageCollection(load_all=True)
    dataToTreat = []
    # Génère une liste de N images, les visualise et affiche leur histo de couleur
    if True:
        # Analyser quelques images pour développer des pistes pour le choix de la représentation
        # N = 6
        # im_list = images.get_samples(N)

        images.generateRepresentation([])
        all_representations = images.representation_coast + images.representation_forest + images.representation_street
        images.

    # Bayes Classifier
    if True:
        print("Rien")
    # ML Classification
    if True:
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
