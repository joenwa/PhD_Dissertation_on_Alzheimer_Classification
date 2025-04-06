# -*- coding: utf-8 -*-
"""
Created on Mon Mar 31 02:23:57 2025

@author: Chukwudumebi
"""

from sklearn.utils.class_weight import compute_class_weight
import numpy as np

def compute_weights(train_gen):
    """
    Computes class weights to handle dataset imbalance.

    Args:
        train_gen: Training data generator.

    Returns:
        class_weights (dict): Dictionary of class weights.
    """
    labels = np.array(train_gen.classes)

    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(labels),
        y=labels
    )

    return {i: weight for i, weight in enumerate(class_weights)}

def train_model(model, train_gen, val_gen, epochs):
    """
    Trains the model with class weights.

    Args:
        model: The compiled CNN model.
        train_gen: Training data generator.
        val_gen: Validation data generator.
        epochs: Number of epochs.

    Returns:
        history: Training history object.
    """
    class_weights = compute_weights(train_gen)

    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs,
        class_weight=class_weights
    )

    model.save("alzheimer_model.h5")
    print("✅ Model training completed!")

    return history  # ✅ Ensure history is returned
