import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class Network(nn.Module):
    
    def __init__(self):
        super().__init__()
        
        #############################
        # Initialize your network
        #############################
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.vgg16 = models.vgg16(pretrained=True)
        self.fc1 = nn.Linear(1000, 8)

        
    def forward(self, x):
        
        #############################
        # Implement the forward pass
        #############################
        self.vgg16.to(self.device)

        # freeze convolution weights
        for param in self.vgg16.features.parameters():
            param.requires_grad = False
        x = self.vgg16(x)
        x = self.fc1(x)
        return x
    
    def save_model(self):
        
        #############################
        # Saving the model's weitghts
        # Upload 'model' as part of
        # your submission
        # Do not modify this function
        #############################
        
        torch.save(self.state_dict(), 'model')

