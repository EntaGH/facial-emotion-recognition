import torch
import torch.nn as nn
import torch.nn.functional as F
from dataset import emotional_classes
class CNN(nn.Module):
    def __init__(self, input_size = 48, num_classes = len(emotional_classes)):
        super().__init__()
        self.options = {
            'input_size':input_size,
            'num_classes':num_classes
        }
        self.num_classes = num_classes
        self.conv1 = nn.Sequential(nn.Conv2d(1, 32, 3, bias=False), nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(nn.Conv2d(32, 64, 3, bias=False), nn.ReLU(inplace=True), nn.MaxPool2d(2, 2), nn.Dropout(0.25))
        self.conv3= nn.Sequential(nn.Conv2d(64, 128, 3, bias=False), nn.ReLU(inplace=True))
        self.conv4 = nn.Sequential(nn.Conv2d(128, 128, 3, bias=False), nn.ReLU(inplace=True), nn.MaxPool2d(2, 2), nn.Dropout(0.25))
        self.fc1 = nn.Sequential(nn.Linear(128*9*9, 1024), nn.Dropout(0.5))
        self.fc3 = nn.Sequential(nn.Linear(1024, num_classes))

    def forward(self, input):
        output = self.conv1(input)
        output = self.conv2(output)
        output = self.conv3(output)
        output = self.conv4(output)
        output = output.view(output.size(0), -1)
        output = self.fc1(output)
        output = self.fc3(output)
        return output

        
    def save(self, path: str):
            print("Saving model to ", path)
            torch.save({"model": self.state_dict(), "options": self.options}, path)

    @staticmethod
    def load(path: str):
            print("Loading model from ", path)
            model = CNN(**torch.load(path)["options"])
            model.load_state_dict(torch.load(path)["model"])
            return model


