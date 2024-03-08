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
from skimage.feature import canny
from skimage.transform import hough_line, hough_line_peaks, probabilistic_hough_line
import os
import glob
import random
from enum import IntEnum, auto

from skimage import color as skic
from skimage import io as skiio
from skimage.color import rgb2gray
from skimage import filters
from scipy.signal import convolve2d

import helpers.analysis as an

IMAGE_SIZE = 2**16
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
        self.medianHSV = np.zeros((len(self.image_list), 3))

        self.images_coast = []
        self.images_forest = []
        self.images_street = []

        self.coast_representation_mean = []
        self.coast_representation_std = []
        self.coast_representation_cov = []
        self.coast_representation_eigen = []

        self.forest_representation_mean = []
        self.forest_representation_std = []
        self.forest_representation_cov = []
        self.forest_representation_eigen = []

        self.street_representation_mean = []
        self.street_representation_std = []
        self.street_representation_cov = []
        self.street_representation_eigen = []

        self.representation_coast = np.zeros((len(self.image_list), 6))
        self.representation_forest = np.zeros((len(self.image_list), 6))
        self.representation_street = np.zeros((len(self.image_list), 6))

        # Crée un array qui contient toutes les images
        # Dimensions [980, 256, 256, 3]
        #            [Nombre image, hauteur, largeur, RGB]
        if load_all:
            self.images = np.array([np.array(skiio.imread(image)) for image in self._path])
            self.all_images_loaded = True

        self.labels = []
        for i in range(len(self.image_list)):
            if 'coast' in self.image_list[i]:
                self.labels.append(ImageCollection.imageLabels.coast)
                self.images_coast.append(self.images[i])
            elif 'forest' in self.image_list[i]:
                self.labels.append(ImageCollection.imageLabels.forest)
                self.images_forest.append(self.images[i])
            elif 'street' in self.image_list[i]:
                self.labels.append(ImageCollection.imageLabels.street)
                self.images_street.append(self.images[i])
            else:
                raise ValueError(i)

    def get_samples(self, N):
        return np.sort(random.sample(range(np.size(self.image_list, 0)), N))

    def generateHistogram(self, image, n_bins=256, channel=3):
        # Construction des histogrammes
        # 1 histogram per color channel
        n_channels = channel
        pixel_values = np.zeros((n_channels, n_bins))
        for i in range(n_bins):
            if n_channels == 1:
                pixel_values[0, i] = np.count_nonzero(image == i)
            else:
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
            self.medianHSV = np.median(imageHSV, axis=(0, 1))
        if view:
            plt.figure()
            plt.hist(self.imageHSVhist[:, :, 0], bins=256, color='red', alpha=0.5)
            plt.hist(self.imageHSVhist[:, :, 1], bins=256, color='green', alpha=0.5)
            plt.hist(self.imageHSVhist[:, :, 2], bins=256, color='blue', alpha=0.5)
            plt.title(f'Histogramme des moyennes de canaux de couleur pour chaque image')
            plt.show()

    def grayPixelCount(self, image):
        grayCount = 0
        for i in range(len(image[0])):
            for j in range(len(image[i])):
                sat = image[i][j][1]
                if sat < 40:
                    grayCount = grayCount + 1

        return (grayCount / IMAGE_SIZE)

    def greenPixelCount(self, image):
        greenCount = 0
        # Définir les plages de teinte pour le vert
        lower_green = np.array([42, 40, 40])  # Borne inférieure pour la teinte verte
        upper_green = np.array([120, 255, 255])  # Borne supérieure pour la teinte verte
        for i in range(len(image[0])):
            for j in range(len(image[i])):
                if(image[i][j][0] > lower_green[0] and image[i][j][0] < upper_green[0]):
                    if(image[i][j][1] > lower_green[1] and image[i][j][1] < upper_green[1]):
                        if(image[i][j][2] > lower_green[2] and image[i][j][2] < upper_green[2]):
                            greenCount = greenCount + 1
        return (greenCount / IMAGE_SIZE)
    def redPixelCount(self, image):
        redCount = 0
        # Définir les plages de teinte pour le vert
        lower_red = np.array([42, 40, 40])  # Borne inférieure pour la teinte route
        upper_red = np.array([220, 255, 255])  # Borne supérieure pour la teinte rouge
        for i in range(len(image[0])):
            for j in range(len(image[i])):
                if(image[i][j][0] < lower_red[0] or image[i][j][0] > upper_red[0]):
                    if(image[i][j][1] > lower_red[1] and image[i][j][1] < upper_red[1]):
                        if(image[i][j][2] > lower_red[2] and image[i][j][2] < upper_red[2]):
                            redCount = redCount + 1
        return (redCount / IMAGE_SIZE)
    def bluePixelCount(self, image):
        blueCount = 0
        # Définir les plages de teinte pour le vert
        lower_blue = np.array([120, 40, 40])  # Borne inférieure pour la teinte verte
        upper_blue = np.array([220, 255, 255])  # Borne supérieure pour la teinte verte
        for i in range(len(image[0])):
            for j in range(len(image[i])):
                if(image[i][j][0] > lower_blue[0] and image[i][j][0] < upper_blue[0]):
                    if(image[i][j][1] > lower_blue[1] and image[i][j][1] < upper_blue[1]):
                        if(image[i][j][2] > lower_blue[2] and image[i][j][2] < upper_blue[2]):
                            blueCount = blueCount + 1
        return (blueCount / IMAGE_SIZE)
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


    def generateRepresentation(self, indexes):
        # Création de la représentation
        # ajoute les pourcentages de pixel gris, rouge, vert, bleu
        for i in range(len(self.images_coast)):
            pixelGr, pixelRed, pixelGreen, pixelBlue = self.applyColorFilter(self.images_coast[i])
            angleMedian, angleIQR = self.applyEdgeFilter(self.images_coast[i])

            self.representation_coast[i][0] = pixelGr
            self.representation_coast[i][1] = pixelRed
            self.representation_coast[i][2] = pixelGreen
            self.representation_coast[i][3] = pixelBlue
            self.representation_coast[i][4] = angleMedian
            self.representation_coast[i][5] = angleIQR

        for i in range(len(self.images_forest)):
            pixelGr, pixelRed, pixelGreen, pixelBlue = self.applyColorFilter(self.images_forest[i])
            angleMedian, angleIQR = self.applyEdgeFilter(self.images_forest[i])

            self.representation_forest[i][0] = pixelGr
            self.representation_forest[i][1] = pixelRed
            self.representation_forest[i][2] = pixelGreen
            self.representation_forest[i][3] = pixelBlue
            self.representation_forest[i][4] = angleMedian
            self.representation_forest[i][5] = angleIQR

        for i in range(len(self.images_street)):
            pixelGr, pixelRed, pixelGreen, pixelBlue = self.applyColorFilter(self.images_street[i])
            angleMedian, angleIQR = self.applyEdgeFilter(self.images_street[i])

            self.representation_street[i][0] = pixelGr
            self.representation_street[i][1] = pixelRed
            self.representation_street[i][2] = pixelGreen
            self.representation_street[i][3] = pixelBlue
            self.representation_street[i][4] = angleMedian
            self.representation_street[i][5] = angleIQR
        print("Street done")

    def applyColorFilter(self, image):

        imHSV = skic.rgb2hsv(image)
        imHSV = np.round(imHSV * (256 - 1))
        pixel_pourcentage_gray = self.grayPixelCount(imHSV)
        pixel_pourcentage_green = self.greenPixelCount(imHSV)
        pixel_pourcentage_blue = self.bluePixelCount(imHSV)
        pixel_pourcentage_red = self.redPixelCount(imHSV)

        return pixel_pourcentage_gray, pixel_pourcentage_red, pixel_pourcentage_green, pixel_pourcentage_blue

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
        resultArray = []
        for i in range(len(indexes)):
            if self.all_images_loaded:
                im = self.images[indexes[i]]
            else:
                im = skiio.imread(self.image_folder + os.sep + self.image_list[indexes[i]])
            ax2[i].imshow(im)

    def do_pca_coast(self, representation):
        # Compute average and std of all representation
        # Then compute eighenvalues and eigenvectors
        for i in representation:
            self.coast_representation_mean.append(np.mean(i, axis=0))
            self.coast_representation_std.append(np.std(i, axis=0))
            cov = np.cov(i.T)

    def do_pca_forest(self, representation):
        # Compute average and std of all representation
        # Then compute eighenvalues and eigenvectors
        for i in representation:
            self.forest_representation_mean.append(np.mean(representation[i], axis=0))
            self.forest_representation_std.append(np.std(representation[i], axis=0))

    def do_pca_street(self, representation):
        # Compute average and std of all representation
        # Then compute eighenvalues and eigenvectors
        for i in representation:
            self.street_representation_mean.append(np.mean(representation[i], axis=0))
            self.street_representation_std.append(np.std(representation[i], axis=0))


    def equalizeHistogram(self, indexes):
        if type(indexes) == int:
            indexes = [indexes]

        fig3 = plt.figure()
        ax3 = fig3.subplots(len(indexes), 2)
        for i in range(len(indexes)):
            if self.all_images_loaded:
                im = self.images[indexes[i]]
            else:
                im = skiio.imread(self.image_folder + os.sep + self.image_list[indexes[i]])
            ax3[i, 0].imshow(im)
            image_equalized, _ = an.equalizeHist(im)
            ax3[i, 1].imshow(image_equalized)
            # im = rgb2gray(image_equalized)
            ax3[i, 2].imshow(im, cmap='gray')

    def gaussian_blur(self, image_array, sigma=1):
        # Définir le noyau du filtre gaussien
        kernel_size = 2 * int(3 * sigma) + 1
        gaussian_kernel = np.zeros((kernel_size, kernel_size))
        for i in range(kernel_size):
            for j in range(kernel_size):
                gaussian_kernel[i, j] = np.exp(-((i - kernel_size // 2) ** 2 + (j - kernel_size // 2) ** 2) / (2 * sigma ** 2))
        gaussian_kernel /= np.sum(gaussian_kernel)

        # Appliquer le filtre de convolution avec le noyau gaussien
        filtered_image = convolve2d(image_array, gaussian_kernel, mode='same', boundary='wrap')

        return filtered_image

    # Effectuer un filtre de laplace sur une image
    # Permet de détecter les contours
    def laplace_operator(self, image_array):
        # Normaliser les valeurs de l'image résultante à [0, 255]
        norm_image = (image_array - np.min(image_array)) / (
                np.max(image_array) - np.min(image_array)) * 255
        # Définir le noyau de l'opérateur de Laplace
        laplace_kernel = np.array([[0, 1, 0],
                                   [1, -4, 1],
                                   [0, 1, 0]])
        filtered_image = convolve2d(norm_image, laplace_kernel, mode='same', boundary='symm')
        #normalisez la réponse laplace
        filtered_image = (filtered_image - np.min(filtered_image)) / (
                np.max(filtered_image) - np.min(filtered_image)) * 255
        return filtered_image
    def applyEdgeFilter(self, image):
        # ax5[i, 0].imshow(im)
        image_equalized, _ = an.equalizeHist(image)
        # change image_equalized to uint8
        # ax5[i, 1].imshow(image_equalized)
        # convert to grayscale
        grayImage = rgb2gray(image_equalized)
        # ax5[i, 2].imshow(im, cmap='gray')
        filter_results = self.gaussian_blur(grayImage, sigma=4)
        filter_results = self.laplace_operator(filter_results)
        # ax5[i, 3].imshow(filter_results, cmap='gray')
        # Trouver les lignes prevalentes
        return self.find_lines(filter_results)


    def find_lines(self, image_laplace):
        # Appliquer la détection de contours de Canny sur l'image de Laplace
        edges = canny(image_laplace, sigma=1.5, low_threshold=0.2, high_threshold=5)
        # Trouver les pics dans la transformée de Hough
        lines = probabilistic_hough_line(edges, threshold=4, line_length=12, line_gap=3)
        # defined array
        angleArray = []
        lineArray = []
        # Extraire et plot les lignes
        for line in lines:
            p0, p1 = line
            lineArray.append((p0, p1))
            angle_rad = np.arctan2(p1[1] - p0[1], p1[0] - p0[0])
            angleArray.append(np.degrees(angle_rad) % 360)

        angleMedian = np.median(angleArray)
        angleIQR = np.percentile(angleArray, 75) - np.percentile(angleArray, 25)
        # Normalisation des angles entre 0 et 1
        angleMedian = angleMedian / 360
        angleIQR = angleIQR / 360
        return angleMedian, angleIQR

    def applyFilterUnsharp(self, indexes):
        if type(indexes) == int:
            indexes = [indexes]

        fig4 = plt.figure()
        ax4 = fig4.subplots(len(indexes), 3)
        for i in range(len(indexes)):
            if self.all_images_loaded:
                im = self.images[indexes[i]]
            else:
                im = skiio.imread(self.image_folder + os.sep + self.image_list[indexes[i]])
            ax4[i, 0].imshow(im)
            # convert to grayscale
            im = rgb2gray(im)
            ax4[i, 1].imshow(im, cmap='gray')
            filter_results = filters.unsharp_mask(im, radius=5, amount=2.0)
            ax4[i, 2].imshow(filter_results)


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
        ax = fig.subplots(len(indexes), 2)

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
            imageXYZ = skic.rgb2ycbcr(imageRGB)
            imagePauvre = rgb2gray(imageRGB)

            imagePauvre = np.round(imagePauvre * 359)

            # Number of bins per color channel pour les histogrammes (et donc la quantification de niveau autres formats)
            n_bins = 256

            # Lab et HSV requiert un rescaling avant d'histogrammer parce que ce sont des floats au départ!
            imageLabhist = an.rescaleHistLab(imageLab, n_bins) # External rescale pour Lab
            imageHSVhist = np.round(imageHSV * (360 - 1))  # HSV has all values between 0 and 100

            # Construction des histogrammes
            histvaluesRGB = self.generateHistogram(imageRGB)
            histtvaluesLab = self.generateHistogram(imageLabhist)
            histvaluesHSV = self.generateHistogram(imageHSVhist, n_bins=360)
            histGray = self.generateHistogram(imagePauvre, channel=1)

            # permet d'omettre les bins très sombres et très saturées aux bouts des histogrammes
            skip = 5
            start = skip
            end = 360 - skip

            # affichage des histogrammes
            ax[image_counter, 0].scatter(range(start, end), histvaluesHSV[0, start:end], s=3, c='red')
            ax[image_counter, 0].scatter(range(start, end), histvaluesHSV[1, start:end], s=3, c='green')
            ax[image_counter, 0].scatter(range(start, end), histvaluesHSV[2, start:end], s=3, c='blue')
            ax[image_counter, 0].set(xlabel='intensité', ylabel='comptes')
            # # ajouter le titre de la photo observée dans le titre de l'histogramme
            image_name = self.image_list[indexes[image_counter]]
            # ax[image_counter, 0].set_title(f'histogramme RGB de {image_name}')
            #
            # # 2e histogramme
            # ax[image_counter, 1].scatter(range(start, end), histtvaluesLab[0, start:end], s=3, c='red')
            # ax[image_counter, 1].scatter(range(start, end), histtvaluesLab[1, start:end], s=3, c='green')
            # ax[image_counter, 1].scatter(range(start, end), histtvaluesLab[2, start:end], s=3, c='blue')
            # ax[image_counter, 1].set(xlabel='intensité', ylabel='comptes')
            # ax[image_counter, 1].set_title(f'histogramme Lab de {image_name}')

            # 3e histogramme
            # ax[image_counter, 0].scatter(range(start, end), histGray[0, start:end], s=3, c='black')
            # ax[image_counter, 1].scatter(range(start, end), histvaluesHSV[0, start:end], s=3, c='black')
            # ax[image_counter, 2].scatter(range(start, end), histvaluesHSV[1, start:end], s=3, c='black')
            # plot the points

            ax[image_counter, 0].set(xlabel='intensité', ylabel='comptes')
            # ax[image_counter, 1].set(xlabel='intensité', ylabel='comptes')


            ax[image_counter, 0].set_title(f'histogramme HSV de {image_name}')
            # ax[image_counter, 1].set_title(f'histogramme HSV de {image_name}')

            # return histvaluesRGB, histtvaluesLab, histvaluesHSV

