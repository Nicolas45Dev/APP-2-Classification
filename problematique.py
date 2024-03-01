"""
Script de départ de la problématique
Problématique APP2 Module IA S8
"""

import matplotlib.pyplot as plt
import numpy as np

from helpers.ImageCollection import ImageCollection



#######################################
def problematique_APP2():
    images = ImageCollection()
    # Génère une liste de N images, les visualise et affiche leur histo de couleur
    # TODO: voir L1.E4 et problématique
    if True:
        # TODO L1.E4.3 à L1.E4.5
        # Analyser quelques images pour développer des pistes pour le choix de la représentation
        N = 3
        im_list = images.get_samples(N)
        # angle1, angle2 = images.applyFilterEdges(im_list)
        # images.equalizeHistogram(im_list)

        # images.applyFilterUnsharp(im_list)
        images.images_display(im_list)
        # images.view_histogrammes(im_list)

        vecteur = images.generateRepresentation(im_list)
        np.set_printoptions(precision=4, suppress=True)
        print(vecteur)
    plt.show()


######################################
if __name__ == '__main__':
    problematique_APP2()
