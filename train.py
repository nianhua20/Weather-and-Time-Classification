import os

import torch
import torchvision
from PIL import Image
from matplotlib import pyplot as plt
from tqdm import tqdm
from torch import nn
from torch.utils.tensorboard import SummaryWriter


class Trainer():
    def __init__(self, model, train_loader, valid_loader, valid_dataset, criterion, optimizer, device, model_save_path):
        self.model = model
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.valid_dataset = valid_dataset
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.save_path = model_save_path
        self.log_path = 'log_34'
        if not os.path.exists(self.log_path):
            os.makedirs(self.log_path )
        self.writer = SummaryWriter(self.log_path)

    def train(self):
        self.model.to(self.device)
        self.model.train()
        # Train the model and save the best one based on validation loss
        best_val_loss = float('inf')
        for epoch in range(30):
            running_loss = 0.0
            total_correct = 0.0
            # Use tqdm to display a progress bar during training
            with tqdm(self.train_loader, unit="batch") as tepoch:
                for i, data in enumerate(tepoch):
                    inputs, weather, period, _ = data
                    inputs = inputs.to(self.device)
                    weather = weather.to(self.device)
                    period = period.to(self.device)
                    self.optimizer.zero_grad()
                    pred_weather, pred_period = self.model(inputs)
                    loss = self.criterion(pred_weather, weather) + self.criterion(pred_period, period)
                    loss.backward()
                    self.optimizer.step()
                    # Calculate the number of correct predictions and add it to the total number of correct predictions
                    _, weather_pred = torch.max(pred_weather, 1)
                    _, period_pred = torch.max(pred_period, 1)
                    total_correct += (weather_pred == weather).sum().item() + (period_pred == period).sum().item()
                    running_loss += loss.item()
                    tepoch.set_postfix(loss=loss.item())

            # Calculate the accuracy and write the loss and accuracy to TensorBoard
            accuracy = total_correct / ((i + 1) * self.train_loader.batch_size * 2)
            self.writer.add_scalar('Training Loss', running_loss / len(self.train_loader), epoch)
            self.writer.add_scalar('Training Accuracy', accuracy, epoch)

            # Calculate the validation loss and accuracy
            val_loss, val_accuracy = self.val_test(epoch)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), self.save_path)

    def val_test(self, epoch=0):
        self.model.to(self.device)
        self.model.eval()

        total_loss = 0.0
        total_correct = 0
        total_weather_tp = 0
        total_weather_fp = 0
        total_weather_fn = 0
        total_period_tp = 0
        total_period_fp = 0
        total_period_fn = 0
        with torch.no_grad():
            for data in self.valid_loader:
                inputs, weather, period, _ = data
                inputs = inputs.to(self.device)
                weather = weather.to(self.device)
                period = period.to(self.device)
                pred_weather, pred_period = self.model(inputs)
                loss = self.criterion(pred_weather, weather) + self.criterion(pred_period, period)
                total_loss += loss.item()
                _, weather_pred = torch.max(pred_weather, 1)
                _, period_pred = torch.max(pred_period, 1)
                total_correct += (weather_pred == weather).sum().item() + (period_pred == period).sum().item()

                # Calculate weather precision, recall, and F-score
                weather_tp = ((weather_pred == weather) & (weather == 1)).sum().item()
                weather_fp = ((weather_pred != weather) & (weather == 0)).sum().item()
                weather_fn = ((weather_pred != weather) & (weather == 1)).sum().item()
                total_weather_tp += weather_tp
                total_weather_fp += weather_fp
                total_weather_fn += weather_fn
                weather_precision = weather_tp / (weather_tp + weather_fp + 1e-8)
                weather_recall = weather_tp / (weather_tp + weather_fn + 1e-8)
                weather_fscore = 2 * weather_precision * weather_recall / (weather_precision + weather_recall + 1e-8)

                # Calculate period precision, recall, and F-score
                period_tp = ((period_pred == period) & (period == 1)).sum().item()
                period_fp = ((period_pred != period) & (period == 0)).sum().item()
                period_fn = ((period_pred != period) & (period == 1)).sum().item()
                total_period_tp += period_tp
                total_period_fp += period_fp
                total_period_fn += period_fn
                period_precision = period_tp / (period_tp + period_fp + 1e-8)
                period_recall = period_tp / (period_tp + period_fn + 1e-8)
                period_fscore = 2 * period_precision * period_recall / (period_precision + period_recall + 1e-8)

            total_loss /= len(self.valid_loader)
            accuracy = total_correct / (len(self.valid_dataset) * 2)
            weather_precision = total_weather_tp / (total_weather_tp + total_weather_fp + 1e-8)
            weather_recall = total_weather_tp / (total_weather_tp + total_weather_fn + 1e-8)
            weather_fscore = 2 * weather_precision * weather_recall / (weather_precision + weather_recall + 1e-8)
            period_precision = total_period_tp / (total_period_tp + total_period_fp + 1e-8)
            period_recall = total_period_tp / (total_period_tp + total_period_fn + 1e-8)
            period_fscore = 2 * period_precision * period_recall / (period_precision + period_recall + 1e-8)
            avg_fscore = (weather_fscore + period_fscore) / 2

            # Write the loss, accuracy, and F-score to TensorBoard
            self.writer.add_scalar('Validation Loss', total_loss, epoch)
            self.writer.add_scalar('Validation Accuracy', accuracy, epoch)
            self.writer.add_scalar('Validation Weather F-score', weather_fscore, epoch)
            self.writer.add_scalar('Validation Period F-score', period_fscore, epoch)
            self.writer.add_scalar('Validation Average F-score', avg_fscore, epoch)
            if (epoch + 1) % 10 == 0:
                # Randomly select 10 images from the validation set and write their predicted and actual labels to TensorBoard
                for i in torch.randperm(len(self.valid_dataset))[:10]:
                    data = self.valid_dataset[i]
                    inputs = data[0].unsqueeze(0).to(self.device)
                    pred_weather, pred_period = self.model(inputs)
                    _, weather_pred = torch.max(pred_weather, 1)
                    _, period_pred = torch.max(pred_period, 1)
                    # Create a figure with the image and labels
                    fig, ax = plt.subplots()
                    ax.imshow(data[3].cpu().permute(1, 2, 0))
                    ax.axis('off')
                    ax.set_title(
                        'Predicted Weather: {}---Actual Weather: {}\nPredicted Period: {}---Actual Period: {}'.format(
                            ['Cloudy', 'Sunny', 'Rainy', 'Snowy', 'Foggy'][weather_pred.item()],
                            ['Cloudy', 'Sunny', 'Rainy', 'Snowy', 'Foggy'][data[1]],
                            ['Dawn', 'Morning', 'Afternoon', 'Dusk', 'Night'][period_pred.item()],
                            ['Dawn', 'Morning', 'Afternoon', 'Dusk', 'Night'][data[2]]
                        ))

                    # Add the figure to TensorBoard
                    self.writer.add_figure('Validation Image {}'.format(i + 1), fig, epoch+1)
        return total_loss, accuracy