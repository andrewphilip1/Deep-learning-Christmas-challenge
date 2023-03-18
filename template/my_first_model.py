import torch
import torch.nn as nn
import torch.nn.functional as F


class Network(nn.Module):
    
    def __init__(self):
        super().__init__()
        
        #############################
        # Initialize your network
        #############################
        self.conv1 = nn.Conv2d(3, 100, kernel_size=7,stride=3)
        self.conv2 = nn.Conv2d(100, 20,kernel_size=7,stride=3)
        self.fc1 = nn.Linear(500, 50)
        self.dropout1 = nn.Dropout2d(0.25)
        self.fc2 = nn.Linear(50, 8)
        
    def forward(self, x):
        
        #############################
        # Implement the forward pass
        #############################
        x = self.conv1(x)
        x = F.max_pool2d(x, 2)
        x = F.relu(x)
        
        x = self.conv2(x)
        x = F.max_pool2d(x, 2)
        x = F.relu(x)
        #print(x.shape)
        
        # Run max pooling over x

        # Flatten x with start_dim=1
        #x = torch.flatten(x, 1)
        x = x.view(-1,500)
        # Pass data through fc1
        x = self.fc1(x)
        x = F.relu(x)
        
        x = self.dropout1(x)
        x = self.fc2(x)
                
        return x
    
    def save_model(self):
        
        #############################
        # Saving the model's weitghts
        # Upload 'model' as part of
        # your submission
        # Do not modify this function
        #############################
        
        torch.save(self.state_dict(), 'model')

