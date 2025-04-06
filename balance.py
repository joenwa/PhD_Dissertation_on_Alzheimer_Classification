# -*- coding: utf-8 -*-
"""
Created on Mon Mar 31 02:53:08 2025

@author: Chukwudumebi
"""

import os
import shutil
import numpy as np

def balance_dataset(data_dir):
    """
    Oversamples minority classes by duplicating images.

    Args:
        data_dir (str): Path to dataset directory.
    """
    class_counts = {cls: len(os.listdir(os.path.join(data_dir, cls))) for cls in os.listdir(data_dir)}
    max_count = max(class_counts.values())

    print(f"Original Class Distribution: {class_counts}")

    for cls, count in class_counts.items():
        if count < max_count:
            cls_dir = os.path.join(data_dir, cls)
            images = os.listdir(cls_dir)
            diff = max_count - count  # Number of images to generate

            for i in range(diff):
                img_name = np.random.choice(images)
                src = os.path.join(cls_dir, img_name)
                dest = os.path.join(cls_dir, f"aug_{i}_{img_name}")
                shutil.copy(src, dest)

    print("Dataset balancing completed.")
