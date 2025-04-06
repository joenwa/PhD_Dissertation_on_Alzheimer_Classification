# -*- coding: utf-8 -*-
"""
Created on Mon Mar 31 02:26:25 2025

@author: Chukwudumebi
"""

import tensorflow as tf
from balance import balance_dataset
from data_loader import load_data
from model import build_model
from rnn_model import build_rnn_model
from transfer_models import build_mobilenet, build_resnet, build_vgg16
from trainer import train_model
from evaluate import test_model
from bias_tracker import track_bias
from visualizer import plot_metrics
from config import DATA_DIR, TEST_DATA_DIR, IMG_SIZE, BATCH_SIZE, EPOCHS, MODEL_SAVE_PATH, MODEL_TYPE

# Ensure only one model type is selected
valid_models = {"cnn", "rnn", "mobilenet", "resnet", "vgg16"}
assert MODEL_TYPE in valid_models, f"Invalid MODEL_TYPE: {MODEL_TYPE}. Choose from {valid_models}"

if __name__ == "__main__":
    print("\n📌 Starting Alzheimer Classification Pipeline...\n")

    # Step 1: Balance dataset
    print("📌 Balancing dataset...")
    balance_dataset(DATA_DIR)

    # Step 2: Load training and validation data
    print("📌 Loading dataset...")
    train_gen, val_gen = load_data(DATA_DIR, IMG_SIZE, BATCH_SIZE)

    # Step 3: Choose Model
    print(f"📌 Training {MODEL_TYPE} model...")

    model_map = {
        "rnn": (build_rnn_model, "rnn"),
        "mobilenet": (build_mobilenet, "mobilenet"),
        "resnet": (build_resnet, "resnet"),
        "vgg16": (build_vgg16, "vgg16"),
        "cnn": (build_model, "cnn"),
    }

    build_fn, model_key = model_map[MODEL_TYPE]
    model = build_fn(IMG_SIZE, num_classes=len(train_gen.class_indices))
    model_save_path = MODEL_SAVE_PATH[model_key]

    # Step 4: Train Model
    history = train_model(model, train_gen, val_gen, EPOCHS)

    if history is None:
        raise ValueError("Error: train_model() returned None. Check trainer.py.")

    # Step 5: Save Model
    model.save(model_save_path)
    print(f"✅ Model saved to {model_save_path}")

    # Step 6: Evaluate on Test Data
    print(f"📌 Testing model {MODEL_TYPE} on test dataset...")
    test_accuracy = test_model(model_save_path, TEST_DATA_DIR, IMG_SIZE, BATCH_SIZE)

    # Step 7: Track Bias
    print("📌 Analyzing model bias...")
    test_gen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1.0 / 255).flow_from_directory(
        TEST_DATA_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode="categorical", shuffle=False
    )
    class_accuracies = track_bias(model, test_gen)

    # Step 8: Visualize Training Results
    print("📌 Plotting training metrics...")
    plot_metrics(history)

    print("\n✅ Pipeline execution completed!")

