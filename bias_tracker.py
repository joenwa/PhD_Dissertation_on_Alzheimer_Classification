# -*- coding: utf-8 -*-
"""
Created on Wed Apr  2 17:14:55 2025

@author: new
"""

import numpy as np

def track_bias(model, test_gen):
    """Checks accuracy per class to detect bias"""
    predictions = model.predict(test_gen)
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = test_gen.classes
    class_labels = list(test_gen.class_indices.keys())

    class_accuracies = {}
    for label in class_labels:
        idx = test_gen.class_indices[label]
        correct = np.sum((predicted_classes == idx) & (true_classes == idx))
        total = np.sum(true_classes == idx)
        class_accuracies[label] = correct / total if total > 0 else 0

    print("Class-wise Accuracy:", class_accuracies)
    return class_accuracies
