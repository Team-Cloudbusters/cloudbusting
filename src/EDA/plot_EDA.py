import matplotlib.pyplot as plt

def plot_class_per_image_hist(df_train):
    """
    Function that plots


    Parameters
    ----------
    df_train: DataFrame
         Data Frame with the run length encoded segmentations for each image-label
         pair in the train_images (provided by Kaggle)

    Returns
    -------
    Plots:
        Two histograms: 1) histogram of label patterns per image_id
                        2) histogram of number of labels per image_id

    """
    from get_train_info import get_class_per_image

    # Get the Data Frame with the cloud labels identified per image_id (in the
    # training images)
    df_class_per_image = get_class_per_image(df_train.dropna().Image_Label)

    # Get label patterns per image_id
    class_pattern = df_class_per_image.cloud_label.value_counts()

    fig, ax = plt.subplots(ncols=2, figsize=(9.2, 7))

    # Plot the histogram of label patterns per image_id
    class_pattern.plot.barh(ax=ax[0])
    ax[0].set_title('Number of each label patterns per images')
    ax[0].set_ylabel('Label patterns')
    ax[0].set_xlabel('Number of images')
    ax[0].grid()

    # Get number of labels per image_id
    num_class_per_image = df_class_per_image.cloud_label.apply(lambda x: \
                                x.count(',')+1).value_counts()
    num_class_per_image = num_class_per_image.sort_index()

    # Plot the histogram of number of labels per image_id
    num_class_per_image.plot.barh(ax=ax[1])
    ax[1].set_title('Number of labels per images')
    ax[1].set_ylabel('Number of cloud labels')
    ax[1].set_xlabel('Number of images')
    ax[1].grid()

    plt.show()
