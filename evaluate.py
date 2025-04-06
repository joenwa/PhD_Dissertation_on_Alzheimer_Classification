# -*- coding: utf-8 -*-
"""
Created on Wed Apr  2 17:12:24 2025

@author: Chukwudumebi
"""

import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from config import TEST_DATA_DIR, IMG_SIZE, BATCH_SIZE

import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from config import TEST_DATA_DIR, IMG_SIZE, BATCH_SIZE

def test_model(model_path, test_data_dir, img_size, batch_size):
    """
    Load the trained model and evaluate it on the test dataset.

    Parameters:
    - model_path (str): Path to the saved model file.
    - test_data_dir (str): Directory containing test dataset.
    - img_size (tuple): Image dimensions (height, width).
    - batch_size (int): Batch size for evaluation.

    Returns:
    - test_accuracy (float): Accuracy of the model on the test dataset.
    """
    print(f"📌 Loading model from {model_path} for testing...")
    model = load_model(model_path)

    # Load test dataset
    test_datagen = ImageDataGenerator(rescale=1.0 / 255)
    test_generator = test_datagen.flow_from_directory(
        test_data_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=False
    )

    # Evaluate model on test data
    print("📌 Evaluating model on test dataset...")
    loss, accuracy = model.evaluate(test_generator, verbose=1)
    print(f"✅ Test Accuracy: {accuracy * 100:.2f}%")

    return accuracy


