import read_files
import get_cluster_centres
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
# from itertools import repeat, chain
import itertools


def get_images_and_labels_1(img_dict):
    img_labels = []
    img_list = []
    dict_len = len(img_dict)
    for key in img_dict.keys():
        for des in img_dict[key]:
            img_labels.append(key)

            # extend label 50 times
            # img_labels.extend(itertools.repeat(key, 50))

            img_des = des
            C = get_cluster_centres.get_cluster_centres(img_des, 50)

            img_list.append(C)

    img_labels_np = np.asarray(img_labels)
    img_list_np = np.asarray(img_list)

    (x, m, n) = img_list_np.shape

    img_list_np_reshaped = img_list_np.reshape(x, (m, n))
                
    return (img_labels_np, img_list_np_reshaped)


    """
        Idea is to vshape the numpy array, each time we add a new image's 
        descriptor list.

    """


def get_images_and_labels(img_dict):

    img_labels = []
    img_list = []

    for key in img_dict.keys():
            for img in img_dict[key]:
                
                img_labels.append(key)
                img_list.append(img)

    return (img_list,  img_labels)
