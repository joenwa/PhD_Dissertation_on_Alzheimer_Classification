# -*- coding: utf-8 -*-
"""
Created on Wed Apr  2 17:42:32 2025

@author: new
"""

# Configuration file for the project

# Data Directories
DATA_DIR = "dataset/AugmentedAlzheimerDataset"
TEST_DATA_DIR = "dataset/OriginalDataset" 

# Image Processing
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10

# Model Selection (Ensure only one is True)
USE_VGG = True
USE_RNN = False
USE_MOBILENET = False
USE_RESNET = False
USE_CNN = False  # Default model

# Ensure only one model is selected
assert sum([USE_RNN, USE_MOBILENET, USE_RESNET, USE_CNN, USE_VGG]) == 1, "⚠️ Select exactly one model type."

# Model Paths
MODEL_SAVE_PATH = {
    "cnn": "saved_models/cnn_model.h5",
    "rnn": "saved_models/rnn_model.h5",
    "vgg16": "saved_models/vgg_model.h5",
    "mobilenet": "saved_models/mobilenet_model.h5",
    "resnet": "saved_models/resnet_model.h5",
}

# Automatically determine the selected model
if USE_RNN:
    MODEL_TYPE = "rnn"
elif USE_MOBILENET:
    MODEL_TYPE = "mobilenet"
elif USE_RESNET:
    MODEL_TYPE = "resnet"
elif USE_VGG:
    MODEL_TYPE = "vgg16"
else:
    MODEL_TYPE = "cnn" # Default

model_save_path = MODEL_SAVE_PATH[MODEL_TYPE.lower()]


