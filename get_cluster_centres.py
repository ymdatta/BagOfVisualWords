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
    """
        Given an image, convert it into grayscale image, then
        compute SIFT feature descriptors for the image and return
        them.

        Parameters:
                   jpg_img - Input image (can be RGB or gray scale).
        Returns:
                   des - It's a numpy array of shape
                         (Number of keypoints x 128).
    """
    img = jpg_img
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    sift = cv2.xfeatures2d.SIFT_create()
    kp, des = sift.detectAndCompute(gray, None)

    return des
