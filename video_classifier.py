import os
import cv2
import math
import random
import numpy as np
import datetime as dt
import tensorflow as tf
from collections import deque
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from tensorflow.keras.layers import *
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical, plot_model


import config
from utils import *



# A function to construct our model
def create_model():
    image_height, image_width = config.IMG_HEIGHT, config.IMG_WIDTH
    model_output_size = config.NUM_CLASSES

    model = Sequential()

    model.add(Conv2D(filters = 64, kernel_size = (3, 3), activation = 'relu', input_shape = (image_height, image_width, 3)))
    model.add(Conv2D(filters = 64, kernel_size = (3, 3), activation = 'relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size = (2, 2)))
    model.add(GlobalAveragePooling2D())
    model.add(Dense(256, activation = 'relu'))
    model.add(BatchNormalization())
    model.add(Dense(model_output_size, activation = 'softmax'))

    model.summary()

    return model

if __name__ == "__main__":

	## For reproducibility
	seed_constant = config.SEED_CONSTANT
	np.random.seed(seed_constant)
	random.seed(seed_constant)
	tf.random.set_seed(seed_constant)

	check_dataset()


	features, labels = create_dataset()
	one_hot_encoded_labels = to_categorical(labels)
	# print(one_hot_encoded_labels)
	features_train, features_test, labels_train, labels_test = train_test_split(features, one_hot_encoded_labels, test_size = 0.2, shuffle = True, random_state = seed_constant)


	model = create_model()

	print("Model Created Successfully!")

	# Optional - For Debugging
	plot_model(model, to_file = 'model_structure_plot.png', show_shapes = True, show_layer_names = True)

	early_stopping_callback = EarlyStopping(monitor = 'val_loss', patience = 10, mode = 'min', restore_best_weights = True)

	model.compile(loss = 'categorical_crossentropy', optimizer = 'Adam', metrics = ["accuracy"])

	model_training_history = model.fit(x = features_train, y = labels_train, epochs = config.NUM_EPOCHS, batch_size = config.BATCH_SIZE , shuffle = True, validation_split = 0.2, callbacks = [early_stopping_callback])

	model_evaluation_history = model.evaluate(features_test, labels_test)


	date_time_format = '%Y_%m_%d__%H_%M'
	current_date_time_dt = dt.datetime.now()
	current_date_time_string = dt.datetime.strftime(current_date_time_dt, date_time_format)
	model_evaluation_loss, model_evaluation_accuracy = model_evaluation_history
	model_name = f'Model_Date_Time_{current_date_time_string}_Loss_{model_evaluation_loss:.2f}_Accuracy_{model_evaluation_accuracy:.2f}.h5'

	# Saving our trained Model
	model.save(model_name)

	plot_metric(model_training_history, 'loss', 'val_loss', 'Total Loss vs Total Validation Loss')
	plot_metric(model_training_history, 'accuracy', 'val_accuracy', 'Total Accuracy vs Total Validation Accuracy')