"""
file: bdutils.py
Files called: bdaugmenter.py
Purpose: the file defines the get_class_weights function, get_sample_counts Functions
         and the custom_standardize funcion. The get_class_weights Function
          will be used in training to account for the imbalance in the dataset.
          get_sample counts function will give the total and class-wise positive
          sample count of a dataset.
credits: based on brucechou1983/CheXNet-Keras
"""

import numpy as np
import os
import pandas as pd

from skimage.transform import resize
from keras import Model
from keras.utils import multi_gpu_model


# counting the number of samples
def get_sample_counts(output_dir, dataset, class_names):

    df = pd.read_csv(os.path.join(output_dir, f"{dataset}.csv"))
    total_count = df.shape[0]
    labels = df[class_names].to_numpy()
    positive_counts = np.sum(labels, axis=0)
    class_positive_counts = dict(zip(class_names, positive_counts))
    return total_count, class_positive_counts
    
# to compensate for the presence of various classes in the dataset
def get_class_weights(total_counts, class_positive_counts, multiply):

    def get_single_class_weight(pos_counts, total_counts):
        denominator = (total_counts - pos_counts) * multiply + pos_counts
        return {
            0: pos_counts / denominator,
            1: (denominator - pos_counts) / denominator,
        }

    class_names = list(class_positive_counts.keys())
    label_counts = np.array(list(class_positive_counts.values()))
    class_weights = []
    for i, class_name in enumerate(class_names):
        class_weights.append(get_single_class_weight(label_counts[i], total_counts))

    return class_weights
