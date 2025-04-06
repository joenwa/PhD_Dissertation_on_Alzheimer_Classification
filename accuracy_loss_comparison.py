# -*- coding: utf-8 -*-
"""
Created on Sun Apr  6 07:32:08 2025

@author: Chukwudumebi
"""

import matplotlib.pyplot as plt
import numpy as np

# Data
models = ['CNN', 'RNN', 'MobileNetV2']
train_accuracies = [0.7269, 0.5764, 0.7292]
test_accuracies = [0.7230, 0.4841, 0.4950]
train_losses = [0.6079, 0.9327, 0.6074]
test_losses = [0.6297, 1.0504, 1.4975]
training_times = [9999 + 9*3300, 9711 + 9*5500, 4113 + 9*1900]  # approximate total times

# Bias (calculated as standard deviation of class-wise accuracies for each model)
biases = [
    np.std([0.8621875, 0.905, 0.476875, 0.648125]),
    np.std([0.5046875, 0.3925, 0.726875, 0.3125]),
    np.std([0.3659375, 0.5621875, 0.9534375, 0.0984375])
]

# Bar Chart Comparison
x = np.arange(len(models))
width = 0.2

fig, ax = plt.subplots(figsize=(12, 7))
bar1 = ax.bar(x - 1.5*width, train_accuracies, width, label='Train Acc')
bar2 = ax.bar(x - 0.5*width, test_accuracies, width, label='Test Acc')
bar3 = ax.bar(x + 0.5*width, train_losses, width, label='Train Loss')
bar4 = ax.bar(x + 1.5*width, test_losses, width, label='Test Loss')

ax.set_xlabel('Models')
ax.set_title('Accuracy and Loss Comparison')
ax.set_xticks(x)
ax.set_xticklabels(models)
ax.legend()
ax.grid(True)

plt.tight_layout()
plt.show()

# Bias Chart
fig2, ax2 = plt.subplots(figsize=(8, 5))
ax2.bar(models, biases, color='orange')
ax2.set_title('Model Bias (Std of Class-wise Accuracy)')
ax2.set_ylabel('Bias (Standard Deviation)')
ax2.grid(True)

plt.tight_layout()
plt.show()

# Training Time Chart
fig3, ax3 = plt.subplots(figsize=(8, 5))
ax3.bar(models, training_times, color='green')
ax3.set_title('Training Time Comparison (in seconds)')
ax3.set_ylabel('Time (s)')
ax3.grid(True)

plt.tight_layout()
plt.show()
