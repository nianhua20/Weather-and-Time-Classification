import json
import os

import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
from tqdm import tqdm

from model import WeatherTimeModel

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = WeatherTimeModel().to(device)

# Load the best model and evaluate it on the test set
model.load_state_dict(torch.load('./saved_models_18/model.pth'))
model.eval()

# Define the transform to be applied to each image
img_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Create a list to store the annotations
annotations = []
annotations1 = []
test_data_path = "data/test_dataset/test_images"

# Loop through each image in the test set folder
for filename in os.listdir(test_data_path):
    # Open the image and apply the transform
    image = Image.open(os.path.join(test_data_path, filename))
    image = img_transform(image)
    # Make a prediction using the trained model
    with torch.no_grad():
        inputs = image.unsqueeze(0).to(device)
        pred_weather, pred_period = model(inputs)
        _, weather_pred = torch.max(pred_weather, 1)
        _, period_pred = torch.max(pred_period, 1)
    # Create a dictionary to store the annotation for this image
    annotation = {
        "filename": os.path.join("test_images", filename),
        "period": ['Dawn', 'Morning', 'Afternoon', 'Dusk', 'Night'][period_pred.item()],
        "weather": ['Cloudy', 'Sunny', 'Rainy', 'Snowy', 'Foggy'][weather_pred.item()]
    }
    annotation1 = {
        "filename": filename,
        "period": ['Dawn', 'Morning', 'Afternoon', 'Dusk', 'Night'][period_pred.item()],
        "weather": ['Cloudy', 'Sunny', 'Rainy', 'Snowy', 'Foggy'][weather_pred.item()]
    }
    # Add the annotation to the list
    annotations.append(annotation)
    annotations1.append(annotation1)

# Create a dictionary to store the annotations list
data = {"annotations": annotations}
# import matplotlib.pyplot as plt

# # Loop through each annotation and display the image with annotations
# for annotation in annotations1:
#     # Load the image
#     image_path = os.path.join(test_data_path, annotation["filename"])
#     image = Image.open(image_path)

#     # Create a figure and axis object
#     fig, ax = plt.subplots()

#     # Display the image
#     ax.imshow(image)

#     # Add the weather and time type annotations
#     ax.text(10, -20, f"Weather: {annotation['weather']}", color='white', fontsize=12, bbox=dict(facecolor='blue', alpha=0.5))
#     ax.text(500, -20, f"Time: {annotation['period']}", color='white', fontsize=12, bbox=dict(facecolor='blue', alpha=0.5))

#     # Show the plot
#     plt.show()

# Serialize the dictionary to a JSON string and write it to a file
with open("annotations.json", "w") as f:
    json.dump(data, f, indent=4)