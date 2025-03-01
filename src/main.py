import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import DataLoader, Dataset
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

def find_closest_match(base_name, image_dict):
    possible_matches = list(image_dict.keys())
    closest_match = difflib.get_close_matches(base_name, possible_matches, n=1, cutoff=0.6)
    return image_dict[closest_match[0]] if closest_match else None

class TIFDataset(Dataset):
    def __init__(self, image_dir, labels_df, transform=None):
        self.image_dir = image_dir
        self.labels_df = labels_df
        self.transform = transform
        self.image_map = self.match_images()

    def match_images(self):
        image_files = glob.glob(os.path.join(self.image_dir, "*.jpg"))
        image_dict = {os.path.basename(img).rsplit(".", 1)[0]: img for img in image_files}
    
        matched_images = []
    
        for _, row in self.labels_df.iterrows():
            base_name = row["image"]
            matched_img = find_closest_match(base_name, image_dict)
    
            if matched_img:
                matched_images.append((matched_img, row["label"]))
    
        return matched_images
        
    def __len__(self):
        # Use the length of the matched images list instead of the original dataframe
        return len(self.image_map)
    
    def __getitem__(self, idx):
        img_path, label = self.image_map[idx]
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)
        
        return image, label

def get_data_loaders(dataset_path, batch_size=64):
    labels_file = os.path.join(dataset_path, "TIF_labels.xlsx")
    image_dir = os.path.join(dataset_path, "images")
    labels_df = pd.read_excel(labels_file)
    
    label_encoder = LabelEncoder()
    labels_df["label"] = label_encoder.fit_transform(labels_df.iloc[:, 1])
    
    train_df, test_df = train_test_split(labels_df, test_size=0.1, random_state=42, stratify=labels_df["label"])
    train_df, val_df = train_test_split(train_df, test_size=0.1, random_state=42, stratify=train_df["label"])
    
    transform = transforms.Compose([
        transforms.Resize((48, 48)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    train_dataset = TIFDataset(image_dir, train_df, transform=transform)
    val_dataset = TIFDataset(image_dir, val_df, transform=transform)
    test_dataset = TIFDataset(image_dir, test_df, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    return train_loader, val_loader, test_loader, label_encoder

def train(model, train_loader, val_loader, device, output_dir, epochs=10):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    train_losses, val_losses, train_accuracies, val_accuracies = [], [], [], []
    log_file = os.path.join(output_dir, "training_log.txt")
    
    with open(log_file, "w") as log:
        for epoch in range(epochs):
            model.train()
            running_loss = 0.0
            correct, total = 0, 0
            
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            
            train_acc = 100 * correct / total
            val_acc, val_loss = evaluate(model, val_loader, criterion, device)
            
            train_losses.append(running_loss / len(train_loader))
            val_losses.append(val_loss)
            train_accuracies.append(train_acc)
            val_accuracies.append(val_acc)
            
            log.write(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader):.4f}, Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%\n")
            print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader):.4f}, Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%")
    
    # Save metrics
    metrics = {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "train_accuracies": train_accuracies,
        "val_accuracies": val_accuracies
    }
    with open(os.path.join(output_dir, "metrics.pkl"), "wb") as f:
        pickle.dump(metrics, f)
    
    # Plot accuracy and loss curves
    plot_training_curves(metrics, output_dir)
    return metrics

def evaluate(model, loader, criterion, device):
    model.eval()
    correct, total, running_loss = 0, 0, 0.0
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    val_acc = 100 * correct / total
    avg_loss = running_loss / len(loader)
    return val_acc, avg_loss, np.array(all_preds), np.array(all_labels)

def plot_training_curves(metrics, output_dir):
    plt.figure()
    plt.plot(metrics['train_losses'], label='Train Loss')
    plt.plot(metrics['val_losses'], label='Validation Loss')
    plt.legend()
    plt.title("Loss Curves")
    plt.savefig(os.path.join(output_dir, "loss_curves.jpg"))
    plt.close()
    
    plt.figure()
    plt.plot(metrics['train_accuracies'], label='Train Accuracy')
    plt.plot(metrics['val_accuracies'], label='Validation Accuracy')
    plt.legend()
    plt.title("Accuracy Curves")
    plt.savefig(os.path.join(output_dir, "accuracy_curves.jpg"))
    plt.close()

def plot_confusion_matrix(cm, classes, output_dir, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    
    fmt = 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
    
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "confusion_matrix.jpg"))
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Train ResNet-18 on a custom emotion dataset.")
    parser.add_argument("--dataset_name", type=str, required=True, help="Name of the dataset")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the dataset")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save model and results")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Unpack all returned values from get_data_loaders
    train_loader, val_loader, test_loader, label_encoder = get_data_loaders(args.dataset_path, args.batch_size)
    
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 7)  # Modify output layer for 7 classes
    model = model.to(device)
    
    print(f"Training ResNet-18 on {args.dataset_name} dataset...")
    metrics = train(model, train_loader, val_loader, device, args.output_dir, args.epochs)
    
    model_save_path = os.path.join(args.output_dir, f"{args.dataset_name}_resnet18.pth")
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved as {model_save_path}")
    
    test_acc, test_loss, test_preds, test_labels = evaluate(model, test_loader, nn.CrossEntropyLoss(), device)
    print(f"Test Accuracy: {test_acc:.2f}%, Test Loss: {test_loss:.4f}")
    cm = confusion_matrix(test_labels, test_preds)
    plot_confusion_matrix(cm, label_encoder.classes_, args.output_dir)

if __name__ == "__main__":
    main()
