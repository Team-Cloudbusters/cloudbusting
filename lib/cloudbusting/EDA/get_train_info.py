from collections import Counter
import pandas as pd
import re

def get_class_per_image(arr):
    """
    Function that returns the classified (cloud) labels per images in the
    training set of (cloud) images.

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
    >>> arr = pd.DataFrame({'A': ["001.jpg_Fish", "001.jpg_Sugar", "002.jpg_Flower"]})
    >>> get_class_per_image(arr.A)
              cloud_label
    image_id             
    001       Fish, Sugar
    002            Flower

    """
    image_id = arr.apply(lambda x: str.split(x, '.')[0])
    class_per_entries = arr.apply(lambda x: str.split(x, '_')[1])
    df_entries = pd.DataFrame({'image_id': image_id, 'cloud_label': class_per_entries})
    df_class_per_image = pd.pivot_table(df_entries, index=['image_id'], columns=[],
                                        values=['cloud_label'],
                                        aggfunc = lambda x: ' '.join(x))
    df_class_per_image['cloud_label'] = \
        df_class_per_image.cloud_label.apply(lambda x: x.strip(' ').replace(' ', ', '))
    return df_class_per_image


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
    >>> arr = pd.DataFrame({'A': ["1.jpg_Fish", "1.jpg_Sugar", "2.jpg_Flower"]})
    >>> get_number_per_class_label(arr.A)
    Counter({'Fish': 1, 'Sugar': 1, 'Flower': 1})

    """
    class_per_entries = arr.apply(lambda x: str.split(x, '_')[1])
    number_per_labels = Counter(class_per_entries)
    return number_per_labels


if __name__ == "__main__":
    import doctest
    doctest.testmod()
