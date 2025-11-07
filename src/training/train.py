
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import numpy as np
import pandas as pd
import cv2
from glob import glob
import scipy.io
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, CSVLogger
from model import build_model

global image_h
global image_w

def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

# def load_dataset(path, split=0.1):
#     """ Loading the images and masks """
#     X = sorted(glob(os.path.join(path, "images", "*.jpg")))
#     Y = sorted(glob(os.path.join(path, "masks", "*.png")))

#     """ Spliting the data into training and testing """
#     split_size = int(len(X) * split)

#     train_x, valid_x = train_test_split(X, test_size=split_size, random_state=42)
#     train_y, valid_y = train_test_split(Y, test_size=split_size, random_state=42)

#     return (train_x, train_y), (valid_x, valid_y)


def load_dataset(path, split=0.1):
    """Loading the images and masks"""
    X = sorted(glob(os.path.join(path, "images", "*.jpg")))
    Y = sorted(glob(os.path.join(path, "masks", "*.png")))

    if len(X) == 0 or len(Y) == 0:
        raise ValueError(f"No images or masks found in {path}. Check dataset directory.")

    """Split data into training and validation"""
    train_x, valid_x = train_test_split(X, test_size=split, random_state=42)
    train_y, valid_y = train_test_split(Y, test_size=split, random_state=42)

    return (train_x, train_y), (valid_x, valid_y)


def read_image(path):
    path = path.decode()
    x = cv2.imread(path, cv2.IMREAD_COLOR)
    x = cv2.resize(x, (image_w, image_h))
    x = x/255.0
    x = x.astype(np.float32)
    return x

def read_mask(path):
    path = path.decode()
    x = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    x = cv2.resize(x, (image_w, image_h))
    x = x.astype(np.float32)    ## (h, w)
    x = np.expand_dims(x, axis=-1)  ## (h, w, 1)
    x = np.concatenate([x, x, x, x], axis=-1) ## (h, w, 4)
    return x

def tf_parse(x, y):
    def _parse(x, y):
        x = read_image(x)
        y = read_mask(y)
        return x, y

    x, y = tf.numpy_function(_parse, [x, y], [tf.float32, tf.float32])
    x.set_shape([image_h, image_w, 3])
    y.set_shape([image_h, image_w, 4])
    return x, y

def tf_dataset(X, Y, batch=2):
    ds = tf.data.Dataset.from_tensor_slices((X, Y))
    ds = ds.map(tf_parse).batch(batch).prefetch(10)
    return ds


if __name__ == "__main__":
    """ Seeding """
    np.random.seed(42)
    tf.random.set_seed(42)

    """ Directory for storing files """
    create_dir("files")

    """ Hyperparameters """
    image_h = 256
    image_w = 256
    input_shape = (image_h, image_w, 3)
    batch_size = 1
    lr = 1e-4
    num_epochs = 20

    """ Paths """
    # dataset_path = "C:\\Users\\Sheetal\\Desktop\\Sheeeeetal DG Model\\Background-Removal-using-Deep-Learning\\data\\people_segmentation"
    # # model_path = os.path.join("files", "model.h5")
    # model_path = os.path.join("files", "model.keras")
    # csv_path = os.path.join("files", "data.csv")
    
    """ Paths """
    # project root = Background-Removal-using-Deep-Learning/
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    DATA_DIR = os.path.join(BASE_DIR, "data", "people_segmentation")
    MLRUNS_DIR = os.path.join(BASE_DIR, "mlruns")

    # create mlruns directory
    create_dir(MLRUNS_DIR)

    dataset_path = DATA_DIR

    model_path = os.path.join(MLRUNS_DIR, "model.keras")
    csv_path   = os.path.join(MLRUNS_DIR, "data.csv")


    """ Loading the dataset """
    (train_x, train_y), (valid_x, valid_y) = load_dataset(dataset_path, split=0.2)
    # 🧪 Use a small subset to test pipeline
    train_x = train_x[:10]
    train_y = train_y[:10]
    valid_x = valid_x[:2]
    valid_y = valid_y[:2]
    print(f"Subset sizes -> Train: {len(train_x)} | Valid: {len(valid_x)}")
    # print(f"Train: {len(train_x)}/{len(train_y)} - Valid: {len(valid_x)}/{len(valid_y)}")
    print("")

    """ Dataset Pipeline """
    train_ds = tf_dataset(train_x, train_y, batch=batch_size)
    valid_ds = tf_dataset(valid_x, valid_y, batch=batch_size)

    """ Model """
    model = build_model(input_shape)
    model.compile(
        loss="binary_crossentropy",
        optimizer=tf.keras.optimizers.Adam(lr)
    )

    """ Training """
    callbacks = [
        ModelCheckpoint(model_path, monitor='val_loss', verbose=1, save_best_only=True),
        # ModelCheckpoint(model_path, monitor='val_loss', verbose=1, save_best_only=True, save_format="tf"),
        ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=1e-7, verbose=1),
        CSVLogger(csv_path, append=True),
        EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=False)
    ]

    model.fit(train_ds,
        validation_data=valid_ds,
        epochs=num_epochs,
        callbacks=callbacks
    )
