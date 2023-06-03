import os

import numpy as np
import torch
from torch import nn
from torchvision import models


class WeatherTimeModel(nn.Module):
    def __init__(self):
        super(WeatherTimeModel, self).__init__()
        # Load the pre-trained ResNet18 model
        backbone = models.resnet18(pretrained=False)
        model_weight_path = "pretrained_models/resnet18.pth"
        assert os.path.exists(model_weight_path), "file {} does not exist.".format(model_weight_path)
        backbone.load_state_dict(torch.load(model_weight_path))
        
        num_features = backbone.fc.in_features
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])
        self.weather_fc = nn.Linear(num_features, 5)
        self.period_fc = nn.Linear(num_features, 5)

    def forward(self, x):
        x = self.backbone(x)
        x = x.view(x.size(0), -1)
        weather = self.weather_fc(x)
        period = self.period_fc(x)
        return weather, period
