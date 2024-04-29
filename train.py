import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import seaborn as sns
from tensorflow import keras
from keras import optimizers, layers, metrics
from keras.callbacks import ModelCheckpoint
from sklearn.metrics import confusion_matrix, classification_report, f1_score, precision_score, recall_score, accuracy_score

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# data sampling & cross-validation not included in this script

Data_X1_0 = np.load("train_npy/Data_X1_0.npy")
Data_X1_1 = np.load("train_npy/Data_X1_1.npy")
Data_X2_0 = np.load("train_npy/Data_X2_0.npy")
Data_X2_1 = np.load("train_npy/Data_X2_1.npy")
Data_y_0 = np.load("train_npy/Data_y_0.npy")
Data_y_1 = np.load("train_npy/Data_y_1.npy")
test_X1 = np.load("train_npy/test_X1.npy")
test_X2 = np.load("train_npy/test_X2.npy")
test_y = np.load("train_npy/test_y.npy")

Data_X1 = np.concatenate((Data_X1_0, Data_X1_1), axis=0)
Data_X2 = np.concatenate((Data_X2_0, Data_X2_1), axis=0)
Data_y = np.concatenate((Data_y_0, Data_y_1), axis=0)

from model import get_model
model = get_model()
checkpoint = ModelCheckpoint('.mdl_wts.hdf5', verbose=1, monitor='val_accuracy',save_best_only=True)
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=['accuracy', metrics.AUC()])
batch_size = 64
model.fit([Data_X1, Data_X2], Data_y, epochs = 20, batch_size=batch_size, verbose = 1, validation_split=0.3,
          callbacks=[checkpoint])
model.load_weights(".mdl_wts.hdf5")
model.evaluate([test_X1,test_X2],test_y)
model.predict([test_X1,test_X2])