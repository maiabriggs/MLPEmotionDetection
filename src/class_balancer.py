import tensorflow as tf
from collections import Counter
from imblearn.over_sampling import SMOTE, SVMSMOTE
import xgboost as xgb
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
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from PIL import Image
import glob
import difflib
import random


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
    
    
    def svm_smote_balancer(self):
        X = []
        y = []
        original_shape = self.images[0].shape
        for image, label in self.data:
            X.append(image.numpy().flatten())  
            y.append(label)
        
        X = np.array(X)
        y = np.array(y)
        print("Class distribution before SMOTE: ", Counter(y))
        smote = SVMSMOTE(sampling_strategy='auto', random_state=42, k_neighbors=3)
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
        
        
    
    
    def sgbdt_balancer(self):
        X = []
        y = []
        original_shape = self.images[0].shape
        for image, label in self.data:
            X.append(image.numpy().flatten())  
            y.append(label)
            
        X = np.array(X)
        y = np.array(y)
        
        print("Class distribution before SGBDT: ", Counter(y))
        
        model = xgb.XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
        model.fit(X, y)
        
        class_counts = Counter(y)
        max_samples = max(class_counts.values())
        
        X_resampled, y_resampled = list(X), list(y)
        
        for class_label, count in class_counts.items():
            if count < max_samples:
                X_minority = X[y == class_label]
                y_minority = y[y == class_label]
                num_samples_add = max_samples - count
                x_synthetic = resample(X_minority, n_samples=num_samples_add, random_state=42, replace=True)
                
                #New labels for new samples
                y_synthetic = [class_label] * num_samples_add  
                
                X_resampled.extend(x_synthetic)
                y_resampled.extend(y_synthetic)
        
        print("Class distribution after SGBDT: ", Counter(y_resampled))
    
        #Convert back to image format:
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
    
    def augment_classes(self, target_ratio=1.0):
        class_counts = Counter(self.labels)
        max_class = max(class_counts.values())
        target_count = int(target_ratio * max_class)  # Target count for balancing

        augmentation_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # Small random shifts
            transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
            transforms.ToTensor()
        ])

        augmented_data = self.data.copy()

        for class_label, count in class_counts.items():
            if count < target_count:
                num_to_add = target_count - count
                class_images = [img for img, label in self.data if label == class_label]

                for _ in range(num_to_add):
                    img = random.choice(class_images)
                    img_pil = transforms.ToPILImage()(img)
                    aug_img = augmentation_transform(img_pil)

                    label_tensor = torch.tensor(class_label, dtype=torch.long)
                    augmented_data.append([aug_img, label_tensor])

        print("Class distribution after augmentation:", Counter([label for _, label in augmented_data]))
        return augmented_data

            
        
         
    
    