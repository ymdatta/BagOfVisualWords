"""
    This file(Kmeans_clustering.py) is used to create
    a visual vocabulary from various images.

    This is further used while training a SVM
    classifier.
"""
import os

from sklearn.cluster import KMeans
from sklearn import svm
from sklearn.metrics import confusion_matrix

import read_files
import vocabulary_helpers
import get_data

import pickle


root, categories, files = os.walk("/path/to/images/kmeans/").next()
# ex: root, categories, files = os.walk("/media/mpl1/mpl_hd2/mohan/BOV/images/kmeans/").next()

# print(categories)

print("Reading images for building a vocabulary.")

vocab_dict = read_files.get_image_dict(categories, "kmeans")

print("Reading images completed. :)")

print("Assigning labels to the images.")
[vocab_images, vocab_labels] = get_data.get_images_and_labels(vocab_dict)

print("Generating vocabulary:")
(f_vocab, i_vocab) = vocabulary_helpers.generate_vocabulary(vocab_images)

n_clusters = 1000

print("Generating clusters: ")
kmeans = vocabulary_helpers.generate_clusters(f_vocab, n_clusters)

print("saving kmeans model: ")
filename = "bov_pickle_1000.sav"

pickle.dump(kmeans, open(filename, 'wb'))
print("Model saved to file: " + str(filename))
