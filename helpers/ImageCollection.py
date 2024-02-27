"""
Classe "ImageCollection" pour charger et visualiser les images de la problématique
Membres :
    image_folder: le sous-répertoire d'où les images sont chargées
    image_list: une énumération de tous les fichiers .jpg dans le répertoire ci-dessus
    images: une matrice de toutes les images, (optionnelle, changer le flag load_all du constructeur à True)
    all_images_loaded: un flag qui indique si la matrice ci-dessus contient les images ou non
Méthodes pour la problématique :
    generateRGBHistograms : calcul l'histogramme RGB de chaque image, à compléter
    generateRepresentation : vide, à compléter pour la problématique
Méthodes génériques : TODO JB move to helpers
    generateHistogram : histogramme une image à 3 canaux de couleurs arbitraires
    images_display: affiche quelques images identifiées en argument
    view_histogrammes: affiche les histogrammes de couleur de qq images identifiées en argument
"""

import matplotlib.pyplot as plt
import numpy as np
import os
import glob
import random
from enum import IntEnum, auto

from skimage import color as skic
from skimage import io as skiio

import helpers.analysis as an


class ImageCollection:
    """
    Classe globale pour regrouper les infos utiles et les méthodes de la collection d'images
    """
    class imageLabels(IntEnum):
        coast = auto()
        forest = auto()
        street = auto()

    def __init__(self, load_all=False):
        # liste de toutes les images
        self.image_folder = r"data" + os.sep + "baseDeDonneesImages"
        self._path = glob.glob(self.image_folder + os.sep + r"*.jpg")
        image_list = os.listdir(self.image_folder)
        # Filtrer pour juste garder les images
        self.image_list = [i for i in image_list if '.jpg' in i]

        self.all_images_loaded = False
        self.images = []
        self.histogrammes = []
        self.meanRGB = np.zeros((len(self.image_list), 3))
        self.meanHSV = np.zeros((len(self.image_list), 3))
        self.meanLab = np.zeros((len(self.image_list), 3))
        self.stdRGB = np.zeros((len(self.image_list), 3))
        self.stdHSV = np.zeros((len(self.image_list), 3))
        self.stdLab = np.zeros((len(self.image_list), 3))

        # Crée un array qui contient toutes les images
        # Dimensions [980, 256, 256, 3]
        #            [Nombre image, hauteur, largeur, RGB]
        if load_all:
            self.images = np.array([np.array(skiio.imread(image)) for image in self._path])
            self.all_images_loaded = True

        self.labels = []
        for i in image_list:
            if 'coast' in i:
                self.labels.append(ImageCollection.imageLabels.coast)
            elif 'forest' in i:
                self.labels.append(ImageCollection.imageLabels.forest)
            elif 'street' in i:
                self.labels.append(ImageCollection.imageLabels.street)
            else:
                raise ValueError(i)

    def get_samples(self, N):
        return np.sort(random.sample(range(np.size(self.image_list, 0)), N))

    def generateHistogram(self, image, n_bins=256):
        # Construction des histogrammes
        # 1 histogram per color channel
        n_channels = 3
        pixel_values = np.zeros((n_channels, n_bins))
        for i in range(n_bins):
            for j in range(n_channels):
                pixel_values[j, i] = np.count_nonzero(image[:, :, j] == i)
        return pixel_values

    def generateRGBHistograms(self, view=False):
        """
        Calcule les histogrammes RGB de toutes les images
        """
        # Générer les moyennes de canal de couleur pour chaque image
        for i in range(len(self.image_list)):
            imageRGB = self.images[i]
            self.meanRGB[i] = (np.mean(imageRGB, axis=(0, 1)))
            self.stdRGB[i] = (np.std(imageRGB, axis=(0, 1)))
        if view:
            plt.figure()
            plt.hist(self.meanRGB[:, 0], bins=256, color='red', alpha=0.5)
            plt.hist(self.meanRGB[:, 1], bins=256, color='green', alpha=0.5)
            plt.hist(self.meanRGB[:, 2], bins=256, color='blue', alpha=0.5)
            plt.title(f'Histogramme des moyennes de canaux de couleur pour chaque image')
            plt.show()

    def generateHSVHistograms(self, view=False):
        # Calculer la mayenne de canal de couleur pour chaque image
        for i in range(len(self.image_list)):
            imageRGB = self.images[i]
            imageHSV = skic.rgb2hsv(imageRGB)
            self.imageHSVhist = np.round(imageHSV * (256 - 1))
            self.meanHSV[i] = (np.mean(self.imageHSVhist, axis=(0, 1)))
            self.stdHSV[i] = (np.std(self.imageHSVhist, axis=(0, 1)))
        if view:
            plt.figure()
            plt.hist(self.imageHSVhist[:, :, 0], bins=256, color='red', alpha=0.5)
            plt.hist(self.imageHSVhist[:, :, 1], bins=256, color='green', alpha=0.5)
            plt.hist(self.imageHSVhist[:, :, 2], bins=256, color='blue', alpha=0.5)
            plt.title(f'Histogramme des moyennes de canaux de couleur pour chaque image')
            plt.show()

    def generateLabHistograms(self, view=False):
        # Calculer la mayenne de canal de couleur pour chaque image
        for i in range(len(self.image_list)):
            imageRGB = self.images[i]
            imageLab = skic.rgb2lab(imageRGB)
            self.imageLabhist = an.rescaleHistLab(imageLab, 256)
            self.meanLab[i] = np.mean(self.imageLabhist, axis=(0, 1))
            self.stdLab[i] = np.std(self.imageLabhist, axis=(0, 1))
        if view:
            plt.figure()
            plt.hist(self.meanLab[:, 0], bins=256, color='red', alpha=0.5)
            plt.hist(self.meanLab[:, 1], bins=256, color='green', alpha=0.5)
            plt.hist(self.meanLab[:, 2], bins=256, color='blue', alpha=0.5)
            plt.title(f'Histogramme des moyennes de canaux de couleur pour chaque image')
            plt.show()


    def generateRepresentation(self):
        # produce a ClassificationData object usable by the classifiers
        # TODO L1.E4.8: commencer l'analyse de la représentation choisie
        raise NotImplementedError()

    def images_display(self, indexes):
        """
        fonction pour afficher les images correspondant aux indices
        indexes: indices de la liste d'image (int ou list of int)
        """
        # Pour qu'on puisse traiter 1 seule image
        if type(indexes) == int:
            indexes = [indexes]

        fig2 = plt.figure()
        ax2 = fig2.subplots(len(indexes), 1)
        for i in range(len(indexes)):
            if self.all_images_loaded:
                im = self.images[indexes[i]]
            else:
                im = skiio.imread(self.image_folder + os.sep + self.image_list[indexes[i]])
            ax2[i].imshow(im)

    def generateAllHistograms(self, indexes):
        if type(indexes) == int:
            indexes = [indexes]

        for image_counter in range(len(indexes)):
            imageRGB = skiio.imread(self.image_folder + os.sep + self.image_list[indexes[image_counter]])

            imageLab = skic.rgb2lab(imageRGB)
            imageHSV = skic.rgb2hsv(imageRGB)

            imageLabhist = an.rescaleHistLab(imageLab, 256)  # External rescale pour Lab
            imageHSVhist = np.round(imageHSV * (256 - 1))  # HSV has all values between 0 and 100

            histvaluesRGB = self.generateHistogram(imageRGB)
            histtvaluesLab = self.generateHistogram(imageLabhist)
            histvaluesHSV = self.generateHistogram(imageHSVhist)

            max_red = max(histvaluesRGB[0])
            max_green = max(histvaluesRGB[1])
            max_blue = max(histvaluesRGB[2])

            index_red = np.where(histvaluesRGB[0] == max_red)[0][0]
            index_green = np.where(histvaluesRGB[1] == max_green)[0][0]
            index_blue = np.where(histvaluesRGB[2] == max_blue)[0][0]

            print("Index of the max count of each channel: ", index_red, max_red, index_green, max_green, index_blue, max_blue)

    def view_histogrammes(self, indexes):
        """
        Affiche les histogrammes de couleur de quelques images
        indexes: int or list of int des images à afficher
        """
        # Pour qu'on puisse traiter 1 seule image
        if type(indexes) == int:
            indexes = [indexes]

        fig = plt.figure()
        ax = fig.subplots(len(indexes), 3)

        for image_counter in range(len(indexes)):
            # charge une image si nécessaire
            if self.all_images_loaded:
                imageRGB = self.images[indexes[image_counter]]
            else:
                imageRGB = skiio.imread(
                    self.image_folder + os.sep + self.image_list[indexes[image_counter]])

            # Exemple de conversion de format pour Lab et HSV
            imageLab = skic.rgb2lab(imageRGB)
            imageHSV = skic.rgb2hsv(imageRGB)

            # Number of bins per color channel pour les histogrammes (et donc la quantification de niveau autres formats)
            n_bins = 256

            # Lab et HSV requiert un rescaling avant d'histogrammer parce que ce sont des floats au départ!
            imageLabhist = an.rescaleHistLab(imageLab, n_bins) # External rescale pour Lab
            imageHSVhist = np.round(imageHSV * (n_bins - 1))  # HSV has all values between 0 and 100

            # Construction des histogrammes
            histvaluesRGB = self.generateHistogram(imageRGB)
            histtvaluesLab = self.generateHistogram(imageLabhist)
            histvaluesHSV = self.generateHistogram(imageHSVhist)

            # permet d'omettre les bins très sombres et très saturées aux bouts des histogrammes
            skip = 5
            start = skip
            end = n_bins - skip

            # affichage des histogrammes
            ax[image_counter, 0].scatter(range(start, end), histvaluesRGB[0, start:end], s=3, c='red')
            ax[image_counter, 0].scatter(range(start, end), histvaluesRGB[1, start:end], s=3, c='green')
            ax[image_counter, 0].scatter(range(start, end), histvaluesRGB[2, start:end], s=3, c='blue')
            ax[image_counter, 0].set(xlabel='intensité', ylabel='comptes')
            # ajouter le titre de la photo observée dans le titre de l'histogramme
            image_name = self.image_list[indexes[image_counter]]
            ax[image_counter, 0].set_title(f'histogramme RGB de {image_name}')

            # 2e histogramme
            ax[image_counter, 1].scatter(range(start, end), histtvaluesLab[0, start:end], s=3, c='red')
            ax[image_counter, 1].scatter(range(start, end), histtvaluesLab[1, start:end], s=3, c='green')
            ax[image_counter, 1].scatter(range(start, end), histtvaluesLab[2, start:end], s=3, c='blue')
            ax[image_counter, 1].set(xlabel='intensité', ylabel='comptes')
            ax[image_counter, 1].set_title(f'histogramme Lab de {image_name}')

            # 3e histogramme
            ax[image_counter, 2].scatter(range(start, end), histvaluesHSV[0, start:end], s=3, c='red')
            ax[image_counter, 2].scatter(range(start, end), histvaluesHSV[1, start:end], s=3, c='green')
            ax[image_counter, 2].scatter(range(start, end), histvaluesHSV[2, start:end], s=3, c='blue')
            # plot the points

            ax[image_counter, 2].set(xlabel='intensité', ylabel='comptes')
            ax[image_counter, 2].set_title(f'histogramme HSV de {image_name}')

            # return histvaluesRGB, histtvaluesLab, histvaluesHSV

