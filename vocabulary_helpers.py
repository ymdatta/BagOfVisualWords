import get_cluster_centres
import numpy as np
from sklearn.cluster import KMeans


def generate_vocabulary(img_list):
    """
        Given a list of images, it generates of vocabulary which contains
        descriptors of all the images present in the img_list.

        Parameters:
                    img_list - list containing all the images.
        Returns:
                    (vocab_list, temp_list)
                    temp_list - list, where each element of the list is
                                a numpy array containing all the feature
                                descriptors of the patches present in an
                                image.

                                ex: l[1].shape == (x, 128)
                    vocab_list: a numpy array, which contains all the
                                descriptors present in all the images.

                                vocab_list.shape == (x, 128)

                                where x is the total no of descriptors
                                present in the images.

                                128- size of descriptor.
    """

    temp_list = []
    for img in img_list:

        features = get_cluster_centres.get_sift_feature_descriptors(img)
        temp_list.append(features)

    vocab_list = temp_list[0]

    for ftr in range(1, len(temp_list)):
        vocab_list = np.vstack((vocab_list, temp_list[ftr]))
        # print(vocab_list.shape)

    return (vocab_list, temp_list)

def generate_clusters(vocab, n_clusters=500):
    """
        given vocabulary, and number of clusters in which vocabulary
        is to be divided, applies kmeans algorithm and returns

        Parameters:
                    vocab - vocabulary containing feature descriptors
                    from all the images available for training.

                    n_clusters - Number of clusters for dividing vocab.
        Returns:
                    kmeans - which is further used to predict.
    """

    kmeans = KMeans(n_clusters).fit(vocab)
    return kmeans


def generate_features(temp_list, kmeans, n_clusters):
    """
        given cluster centres(via kmeans) which are obtained from
        all the training images, what this function does is,
        for each image it creates a histogram where x axis is
        nothing but all the clusters(0 to n_clusters) and
        for each descriptor of the image we calculate to which
        cluster it belongs and for each cluster y-axis contains
        no of times a cluster is predicted via kmeans.predict(descriptor).

        This can be further used for training classifier.

        Parameters:
                    temp_list = is the same temp_list from
                                generate_vocabulary.
                    n_clusters = no of clusters in which training
                                vocab is divided into.
                    kmeans = kmeans obtained from generate_clusters.

        Returns:
                    p_array - a numpy array which is of size
                              no_images x no_of_clusters.
    """

    t_length = len(temp_list)
    p_array = np.zeros((1, n_clusters))

    for i in range(0, t_length):

        features = temp_list[i]
        (x1, y1) = features.shape

        # an array for each image
        t_array = np.zeros((1, n_clusters))

        for j in range(0, x1):
            des = features[j]
            des = des.reshape(1, 128)
            ind = kmeans.predict(des)
            t_array[0][ind] += 1

        p_array = np.vstack((p_array, t_array))

    p_array = p_array[1:]
    return p_array
