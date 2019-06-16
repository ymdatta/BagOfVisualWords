import cv2
import os


def get_image_dict(categories, i_type="test"):
    """
        Given categories and whether test/train type images,
        searches in the images directory accordingly and
        returns the images.

        Parameters: categories - list of strings.
                                 (Each string is a category).
                    i_type = specifies which images(i.e test/train)
                             Default is "test"

        Return: returns a dictionary, with keys representing
                categories and its values representing images
                present in the category.
    """

    if i_type == "test" or i_type == "train" or i_type == "kmeans":
        # Run only if i_type is either test/train.
        image_dict = {}
        for (index, category) in list(enumerate(categories)):
            add_images(category, image_dict, index, i_type)
        return image_dict
    else:
        raise ValueError("Image type is not test or train")


def add_images(category, image_dict, index, i_type="test"):
    """
        Given category name, dictionary to which we need to add
        images, index and whether images are test/train type
        returns dictionary with adding images present in the
        category.

        Parameters: category - category name.
                    image_dict - dictionary to which images should be stored.
                    index - key value for dictionary.
                    i_type - either "test"/"train"/"kmeans"

        Returns:
                    Nothing, since image_dict is already modified above.
        """
    temp_path = os.path.join(i_type, category)
    pathname = os.path.join("/path/to/images", temp_path)
    # ex: pathname = os.path.join("/home/ymdatta/BOV/images", temp_path)

    image_dict[index] = []

    print("Category: " + str(category))
    for img in os.listdir(pathname):
        path_temp = os.path.join(pathname, img)
        print(path_temp)
        temp_img = cv2.imread(path_temp)
        image_dict[index].append(temp_img)
    return
