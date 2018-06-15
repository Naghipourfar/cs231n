import numpy as np
import pandas as pd

import tensorflow as tf
import matplotlib.pyplot as plt
import keras.backend as K

"""
    Created by Mohsen Naghipourfar on 6/7/18.
    Email : mn7697np@gmail.com
    Website: http://ce.sharif.edu/~naghipourfar
"""


def svm_loss(y_pred, y_true):  # Also known as Hinge loss function
    '''
        The Hinge Loss function
        This function only care about the answer and difference of the correct label with others for correct prediction doesn't matter.

        Equation:
        Loss_i = sum_{j!=i} (max(0, y_j - y_i + 1))
        Loss = sum_i (Loss_i)

    '''
    loss = K.mean(K.max(0, y_pred - y_true))
    return loss
