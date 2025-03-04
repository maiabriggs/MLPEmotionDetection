import tensorflow as tf
from collections import Counter
from imblearn.over_sampling import SMOTE

data_dir = 'datasets/AffectNet'


dataset = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    labels='inferred',      
    label_mode='int',      
    image_size=(64, 64),    
    batch_size=None         
)

X = []
y = []

for image, label in dataset:
    X.append(image.numpy().flatten())  
    y.append(label.numpy())

import numpy as np
X = np.array(X)
y = np.array(y)

print("Data loaded")
print(f"Shape of X: {X.shape}, Shape of y: {y.shape}")

print("Class distribution before SMOTE:", Counter(y))

smote = SMOTE(sampling_strategy='auto', random_state=42)

X_resampled, y_resampled = smote.fit_resample(X, y)

print("Class distribution after SMOTE:", Counter(y_resampled))

X_resampled = X_resampled.reshape(-1, 64, 64, 3)
print(f"Shape after reshaping: {X_resampled.shape}")