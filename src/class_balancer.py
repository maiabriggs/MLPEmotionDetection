import tensorflow as tf
from collections import Counter
from imblearn.over_sampling import SMOTE
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import DataLoader, Dataset
import numpy as np
import argparse
import os
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from PIL import Image
import glob
import difflib


#Data is in format [[img, label], [img, label], ...]
class Class_Balancer():
    def __init__(self, data, transform=None):
        self.labels = []
        self.images = []
        self.data = data
        self.transform = transform
        for image, label in data:
            self.images.append(image)
            self.labels.append(label)
            
    
    """
    Takes array of images and labels and runs SMOTE on them, returns data in the form:
    [[img, label], [img, label], ...]
    """
    def smote_balancer(self):
        X = []
        y = []
        original_shape = self.images[0].shape
        for image, label in self.data:
            X.append(image.numpy().flatten())  
            y.append(label)
        
        X = np.array(X)
        y = np.array(y)
        print("Class distribution before SMOTE: ", Counter(y))
        smote = SMOTE(sampling_strategy='auto', random_state=42, k_neighbors=3)
        print("Running SMOTE")
        X_resampled, y_resampled = smote.fit_resample(X, y)
        print("Class distribution after SMOTE: ", Counter(y_resampled))
        
        resampled_data = []
        for img, label in zip(X_resampled, y_resampled):
            img = img.reshape(original_shape)
            img_tensor = torch.tensor(img, dtype=torch.float32)
            
            if self.transform:
                img_tensor = img_tensor.squeeze() 
                img_array = img_tensor.permute(1, 2, 0).cpu().numpy() if img_tensor.ndim == 3 else img_tensor.cpu().numpy()
                img_array = (img_array * 255).astype(np.uint8)  
                img = Image.fromarray(img_array)  
                img_tensor = self.transform(img)  

            label_tensor = torch.tensor(label, dtype=torch.long)
            resampled_data.append([img_tensor, label_tensor])
        
        return resampled_data
    
    
        
        
        
            
        
        
        
    
    