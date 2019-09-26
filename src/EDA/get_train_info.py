from collections import defaultdict
import re

def get_number_per_image_id(arr):
    """
    Function that returns the number of entries per image id in the training set
     of (cloud) images, ignoring the missing entries.

    Parameters
    ----------
    arr : array-like
        The Image_Label column of the training.csv (after dropping the missing
        entries, where the entries are str dtype in the form "XXX.jpg_(class_name)".
        Here class_name is {Fish, Flower, Gravel, Sugar}

    Returns
    -------
    number_per_image_id : dict
        Data dictionary with the number of entries per image_id.

    Examples
    --------
    >>> arr1 = pd.Series({'A': ["001.jpg_Fish", "001.jpg_Sugar", "002.jpg_Flower"]})
    >>> get_number_per_image_id(arr1.A)
    defaultdict(int, {'001': 2, '002': 1})

    """
    number_per_image_id = defaultdict(int)
    for image_label in arr:
        image_id = str.split(image_label, '.')[0]
        number_per_image_id[image_id] += 1
    return number_per_image_id


def get_number_per_class_label(arr):
    """
    Function that returns the total number of images per (cloud) class labels,
    ignoring the missing entries.

    Parameters
    ----------
    arr : array-like
        The Image_Label column of the training.csv (after dropping the missing
        entries, where the entries are str dtype in the form "XXX.jpg_(class_name)".
        Here class_name is {Fish, Flower, Gravel, Sugar}

    Returns
    -------
    number_per_labels : dict
        Data dictionary with the total number of images per (cloud) class labels.

    Examples
    --------
    >>> arr1 = pd.Series({'A': ["1.jpg_Fish", "1.jpg_Sugar", "2.jpg_Flower"]})
    >>> get_number_per_image_id_per_class_label(arr1.A)
    {'Fish': 1, 'Flower': 1, 'Gravel': 0, 'Sugar': 1}

    """
    number_per_labels = {"Fish":0, "Flower":0, "Gravel":0, "Sugar":0}
    for image_label in arr:
        for label in number_per_labels.keys():
            if re.search(label, image_label):
                number_per_labels[label] += 1
                break
    return number_per_labels
