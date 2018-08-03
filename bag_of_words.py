import cv2
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn import svm
from sklearn.metrics import confusion_matrix

import read_files
import vocabulary_helpers
import get_data
import plot_data

import pickle






# categories which have more than 100 images: 80 - train 20 -test. 
        # 1. airplanes.
        # 2. chandelier.
        # 3. motorbikes.
        
# categories which have moret than 90 images: 70 - train 20 - test - butterfly.
        # 1. butterlfy.

categories = ['airplanes', 'chandelier', 'motorbikes', 'butterfly', 'revolver', 'spoon']

print("Training Images: ")

print("Readin image files: ")
train_dict = read_files.get_image_dict(categories, "train")
test_dict = read_files.get_image_dict(categories, "test")

print("Parsing image files into categories: ")
[train_imgs, train_labels] = get_data.get_images_and_labels(train_dict)
[test_imgs, test_labels] = get_data.get_images_and_labels(test_dict)

print("Generating vocabulary: ")
(f_vocabulary, i_vocab)= vocabulary_helpers.generate_vocabulary(train_imgs)
(t_f_vocabulary, t_i_vocab)= vocabulary_helpers.generate_vocabulary(test_imgs)

n_clusters = 1000

# kmeans = vocabulary_helpers.generate_clusters(f_vocabulary, n_clusters)
kmeans = pickle.load(open("bov_pickle_1000.sav", 'rb'))

print("generating features: ")

print("Creating feature vectors for test and train images from vocabulary set.")
train_data = vocabulary_helpers.generate_features(i_vocab, kmeans, n_clusters)
test_data = vocabulary_helpers.generate_features(t_i_vocab, kmeans, n_clusters)

print("Applying SVM classifier.")
# SVM Classifier.
clf = svm.SVC()
fitted = clf.fit(train_data, train_labels)
predict = clf.predict(test_data)

print("Actual:")
print(test_labels)

print("predicted:")
print(predict)


#Confusion matrix.
test_labels = np.asarray(test_labels)
cnf_matrix = confusion_matrix(predict, test_labels)
np.set_printoptions(precision = 2)


plot_data.plot_confusion_matrix(cnf_matrix, classes = categories,
                                title='Confusion matrix')
plt.show()
