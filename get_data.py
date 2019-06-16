def get_images_and_labels(img_dict):
    """
        Given an img_dict, creates two lists img_labels and img_list

        Generally img_dict is modelled as to contain label as key
        and the key's value contains a list which has all the
        images present in the category.

        Parameters:
                    img_dict - dictionary with keys containing label category
                               and corresponding values containing images
                               in that category.

        Returns:
                    (img_list, img_labels)

                    img_labels: contains labels of all the images.
                    img_list: contains list of all images.
    """

    img_labels = []
    img_list = []

    for key in img_dict.keys():
            for img in img_dict[key]:

                img_labels.append(key)
                img_list.append(img)

    return (img_list,  img_labels)

# TODO: Changes for imporovement:
    # Do something instead of 2 loops.
