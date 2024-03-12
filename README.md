# APP-2-Classification

**Contenu**
- [APP-2-Classification](#app-2-classification)
  - [Description du projet](#description-du-projet)
  - [Struture du projet](#struture-du-projet)
  - [Calcul pour le prétraitement et représentation](#calcul-pour-le-prétraitement-et-représentation)
  - [Classification](#classification)
  - [Classificateur Bayes](#classificateur-bayes)
  - [Classificateur PPV](#classificateur-ppv)
  - [Classificateur Réseaux de neuronnes](#classificateur-réseaux-de-neuronnes)

## Description du projet
Projet de classification de données avec trois classificateur. Le projet est divisé en trois parties:
- Partie 1: Préparation des données (prétraitement, normalisation, etc.)
- Partie 2: représentation des données (vecteur)
- Partie 3: Classification des données

## Struture du projet
Dans le fichier [problematique.py](problematique.py) se trouve le code les appels pour effectuer toutes les étapes de la classification. Les fonctions nécessaire pour effectuer les étapes de pré-traitement et générer les représentations sont dans [ImageCollection.py](./helpers/ImageCollection.py). Par la suite, les classificateurs se trouve dans le fichier [Classifiers.py](./helpers/Classifiers.py).

## Calcul pour le prétraitement et représentation
La fonction [generateRepresentation()](problematique.py#L23) fait tous les appels pour générer les représentations des images. Par la suite, le programme enregistre les représentations dans un fichier npy. Ainsi, les prochaines exécutions pourrait être réaliser plus rapidement. Dans la focntion de génération de représentation, les calculs sont les suivants: un calcul de pourcentage de pixels grise dans les images [grayPixelCount](./helpers/ImageCollection.py#155), un calcul de quantité de lignes dans l'image moins les lignes qui sont horizontales et verticales [find_lines](./helpers/ImageCollection.py#372) et un calcul de quantité de lignes verticales [find_lines](./helpers/ImageCollection.py#372). Le premier calcul calcul le nombre de pxiel qui se trouve dasn des bornes de couleurs. Dans le cas de la fonction se sont les bornes de gris. Les bornes sont représentées en valeurs HSV. Le deuxième calcul applique plusieurs filtre pour trouver les lignes horizontales et verticales. En premier, une fonction égalise l'image pour augmenter le contraste. Par la suite, une fonction applique un filtre de flou Gausien. Ensuite, un filtre de Sobel est appliqué pour trouver les lignes horizontales et verticales. Finalement, un filtre de Canny est appliqué pour trouver les contours. Le troisième calcul se fait en même quand que le deuxième. Des conditions permettent d'identifier les lignes avec des angles verticales donc 90 degrées.

## Classification
Pour la classification les valeurs ont été normalisé entre 0 et 1. Les données utilisées pour les classificateurs sont les représentations des images sous le format npy. Les fichiers seront chargés lorsqu'un objet de la classe [ClassificationData](./helpers/ClassificationData.py#L1) est instancié. Par la suite, les différentes sont appelés avec les paramètres nécessaires pour effectuer la classification.

## Classificateur Bayes
Le classificateur de Bayes peut prendre en compte les apriori et des coûts pour minimisé les faux positifs. Ces paramètres se nomment: apriori et cost. Les valeurs pour l'apriori doit être un array de une dimension avec la même quantité d'éléments que de classes. Les valeurs pour le coût doit être un array de deux dimensions avec la même quantité de lignes et de colonnes que de classes

## Classificateur PPV
Le classificateur PPV demande seulement un nombre de voisins pour effectuer la classification. Les valeurs pour le nombre de voisins doit être un entier positif.

## Classificateur Réseaux de neuronnes
Le classificateur de réseaux de neuronnes demande un nombre de couches et un nombre de neurones par couche. Les valeurs pour le nombre de couches cachées doit être un entier positif. Les valeurs pour le nombre de neurones par couche doit être un entier positif. De plus, il est nécessaire de spécifier le nombre d'itérations, le taux d'apprentissage et le type d'activation. Les valeurs pour le nombre d'itérations doit être un entier positif. Les valeurs pour le taux d'apprentissage doit être un float positif entre 0 et 1. Les valeurs pour le type d'activation doit être un string avec les valeurs: 'tanh', 'relu' ou 'sigmoid'.