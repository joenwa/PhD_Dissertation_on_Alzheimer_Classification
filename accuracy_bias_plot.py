# -*- coding: utf-8 -*-
"""
Created on Sun Apr  6 07:28:44 2025

@author: Chukwudumebi
"""

import matplotlib.pyplot as plt
import numpy as np

# Data extracted from the logs
models = ['CNN', 'RNN', 'MobileNetV2.0', 'VGG16']
train_accuracies = [0.7269, 0.5764, 0.7292, 0]
test_accuracies = [0.7230, 0.4841, 0.4950, 0]
train_losses = [0.6079, 0.9327, 0.6074, 0]
test_losses = [0.6297, 1.0504, 1.4975, 0]
training_times = [9999 + 9*3400, 9711 + 9*5500, 4113 + 9*1900]  # approximate total seconds for training
bias_scores = [  # Mean difference from 0.5 (neutral) across class-wise accuracy
    np.mean([0.8621875, 0.905, 0.476875, 0.648125]),
    np.mean([0.5046875, 0.3925, 0.726875, 0.3125]),
    np.mean([0.3659375, 0.5621875, 0.9534375, 0.0984375])
]

# Plotting
x = np.arange(len(models))
width = 0.2

fig, ax = plt.subplots(figsize=(12, 7))
rects1 = ax.bar(x - width, train_accuracies, width, label='Train Accuracy')
rects2 = ax.bar(x, test_accuracies, width, label='Test Accuracy')
rects3 = ax.bar(x + width, bias_scores, width, label='Bias (avg acc)')

ax.set_ylabel('Scores')
ax.set_title('Model Comparison: Accuracy and Bias')
ax.set_xticks(x)
ax.set_xticklabels(models)
ax.legend()
plt.ylim(0, 1.1)

# Adding data labels
for rects in [rects1, rects2, rects3]:
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

plt.tight_layout()
plt.show()
