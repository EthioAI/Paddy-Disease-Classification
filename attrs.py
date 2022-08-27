import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
import gc


def load_imgs_from_dataset(label_paths, paths, isTrain=True):
    """
    Load image from dataset
    """
    super_path = "train_images" if isTrain else "test_images"
    if isTrain:
        imgs = np.array(
            [cv2.resize(plt.imread(os.path.join("data", super_path, label_path, path[0])),
                        (120, 160)) for (label_path, path) in zip(label_paths, paths)]
        )
    else:
        imgs = np.array(
            [cv2.resize(plt.imread(os.path.join("data", super_path, path)),
                        (120, 160)) for path in paths]
        )
    return imgs


def split_dataset(X: np.ndarray, y: np.ndarray):
    """
    Splits the dataset into a training and a test set.
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    X_train = load_imgs_from_dataset(y_train, X_train.values)
    X_test = load_imgs_from_dataset(y_test, X_test.values)
    X_valid = load_imgs_from_dataset(y_valid, X_valid.values)

    return X_train, X_test, X_valid, y_train.values.reshape(-1, 1), y_test.values.reshape(-1, 1), y_valid.values.reshape(-1, 1)


def clear_session():
    """
    Clears the current Keras session.
    """
    gc.collect()
    tf.keras.backend.clear_session()
