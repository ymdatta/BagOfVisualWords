import cv2
from sklearn.cluster import KMeans


def get_cluster_centres(jpg_img, k=8):
    img = jpg_img
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    sift = cv2.xfeatures2d.SIFT_create()
    kp, des = sift.detectAndCompute(gray, None)

    [num, size] = des.shape
    # print(num)

    kmeans = KMeans(k)
    C = kmeans.cluster_centers_

    return C


def get_sift_feature_descriptors(jpg_img):
    img = jpg_img
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    sift = cv2.xfeatures2d.SIFT_create()
    kp, des = sift.detectAndCompute(gray, None)

    return des
