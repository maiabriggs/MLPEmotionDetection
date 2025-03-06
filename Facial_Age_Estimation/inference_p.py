import torch
import os
import glob
import csv
from PIL import Image
import torchvision.transforms as T
from model import AgeEstimationModel
from config import config
import datetime

def inference(model, image_path):
    model.eval()
    with torch.no_grad():
        image = Image.open(image_path).convert('RGB')
        transform = T.Compose([
            T.Resize((config['img_width'], config['img_height'])),
            T.ToTensor(),
            T.Normalize(mean=config['mean'], std=config['std'])
        ])
        input_data = transform(image).unsqueeze(0).to(config['device'])
        outputs = model(input_data)  # Forward pass through the model
        
        # Extract the age estimation value from the output tensor
        age_estimation = outputs.item()
        return age_estimation

# Define the folder path containing the input images
image_folder = config['image_folder_path']

# Get list of all image files in the folder (filtering by common image extensions)
image_files = glob.glob(os.path.join(image_folder, '*'))
image_files = [f for f in image_files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

# Load the model from the latest checkpoint
checkpoint_dir = config.get('checkpoint_dir', "/Users/saravut_lin/EDINBURGH/Semester_2/MLP/Facial_Age_estimation_PyTorch/checkpoints")
checkpoint_files = glob.glob(os.path.join(checkpoint_dir, 'epoch-*-loss_valid-*.pt'))
latest_checkpoint = max(checkpoint_files, key=os.path.getctime)

model = AgeEstimationModel(
    input_dim=3,
    output_nodes=1,
    model_name=config['model_name'],
    pretrain_weights='IMAGENET1K_V2'
).to(config['device'])

# Load the model state (mapping to CPU for compatibility, then move to device if needed)
model.load_state_dict(torch.load(latest_checkpoint, map_location=torch.device('cpu')))

# List to store results (each element is [image_filename, predicted_age])
results = []

for image_path in image_files:
    predicted_age = inference(model, image_path)
    results.append([f"surprise/{os.path.basename(image_path)}", predicted_age])
    print(f"Processed {os.path.basename(image_path)}: Age {predicted_age:.2f}")

# Create a new CSV filename in the output folder with a timestamp
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
output_folder = config['output_path_test']
output_csv = os.path.join(output_folder, f"results_{timestamp}.csv")

with open(output_csv, mode='w', newline='') as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(['Image', 'Age'])
    writer.writerows(results)

print(f"CSV file saved to {output_csv}")