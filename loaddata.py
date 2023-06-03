import json
import os
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt


from torch.utils.data import Dataset, DataLoader

# Define a dictionary to map the weather and period labels to integers
weather_dict = {'Cloudy': 0, 'Sunny': 1, 'Rainy': 2, 'Snowy': 3, 'Foggy': 4}
period_dict = {'Dawn': 0, 'Morning': 1, 'Afternoon': 2, 'Dusk': 3, 'Night': 4}


# Modify the __getitem__ method to encode the weather and period labels as integers
class WeatherDataset(Dataset):
    def __init__(self, data_path):
        self.data_path = data_path
        self.json_path = os.path.join(data_path, 'train.json')
        with open(self.json_path, 'r') as f:
            self.data = json.load(f)['annotations']
        self.image_path = [os.path.join(data_path, d['filename']).replace('\\', '/') for d in self.data]
        self.image_period = [d['period']  for d in self.data]
        self.image_weather = [d['weather']  for d in self.data]

        self.data_transforms = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.oriImage_transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.image_path[idx]
        ori_img = Image.open(img_path)
        img = self.data_transforms(ori_img)
        ori_img = self.oriImage_transforms(ori_img)
        weather = self.data[idx]['weather']
        period = self.data[idx]['period']

        # Encode the weather and period labels as integers
        weather = weather_dict[weather]
        period = period_dict[period]

        return img, weather, period, ori_img
    
    def show_weatherAndperiod_count(self):
        # Count the number of images for each weather and period
        weather_count = [0] * len(weather_dict)
        period_count = [0] * len(period_dict)

        for i in range(len(self.data)):
            weather_count[weather_dict[self.image_weather[i]]] += 1
            period_count[period_dict[self.image_period[i]]] += 1

        # Print the counts for each weather and period
        print("Weather counts:")
        for weather, count in weather_dict.items():
            print(f"{weather}: {weather_count[count]}")

        print("Period counts:")
        for period, count in period_dict.items():
            print(f"{period}: {period_count[count]}")
            import matplotlib.pyplot as plt

        # Create a bar chart for weather counts
        plt.bar(weather_dict.keys(), weather_count)
        plt.title('Weather Counts')
        plt.xlabel('Weather')
        plt.ylabel('Count')
        plt.show()

        # Create a bar chart for period counts
        plt.bar(period_dict.keys(), period_count)
        plt.title('Period Counts')
        plt.xlabel('Period')
        plt.ylabel('Count')
        plt.show()


