# -*- coding: utf-8 -*-
"""
Created on Wed Apr  2 17:14:22 2025

@author: Chukwudumebi
"""

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, LSTM, Reshape

def build_rnn_model(img_size, num_classes):
    model = Sequential([
        Conv2D(64, (3,3), activation='relu', padding='same', input_shape=(img_size[0], img_size[1], 3)),
        MaxPooling2D(2,2),
        Conv2D(128, (3,3), activation='relu', padding='same'),
        MaxPooling2D(2,2),
        Flatten(),
        Reshape((1, -1)),  # Convert to sequence format for LSTM
        LSTM(64, return_sequences=False),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')  
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model
