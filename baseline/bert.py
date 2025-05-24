import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
from config import *

class SentimentClassifier(nn.Module):
    """
    Simple sentiment classifier that takes a representation (e.g., CLS token) as input 
    and outputs probability scores for different sentiment polarities.
    """
    def __init__(self, hidden_size=hidden_size):
        """
        Initialize the SentimentClassifier.
        
        Args:
            hidden_size (int): The size of the hidden layer in the classifier.
        """
        super(SentimentClassifier, self).__init__()
        
        # Define a sequential model for the classifier
        self.classifier = nn.Sequential(
            nn.Linear(int(hidden_dim), hidden_size),  # Linear layer from hidden_dim to hidden_size
            nn.Tanh(),  # Tanh activation function
            nn.Linear(hidden_size, num_polarities)  # Linear layer from hidden_size to num_polarities
            # Note: Softmax is not applied here because it will be included in the loss function
        )
        
        # Initialize weights of the classifier
        self.init_weight(self.classifier)

    def init_weight(self, seq):
        """
        Initialize the weights of the given sequential model.
        
        Args:
            seq (nn.Sequential): The sequential model to initialize.
        """
        for layer in seq:
            if isinstance(layer, nn.Linear):
                nn.init.uniform_(layer.weight, a=-0.1, b=0.1)  # Uniform initialization for weights in range [-0.1, 0.1]
                nn.init.constant_(layer.bias, 0)  # Initialize biases to 0

    def forward(self, representation: torch.Tensor):
        """
        Forward pass through the classifier.
        
        Args:
            representation (torch.Tensor): The input tensor 
        Returns:
            torch.Tensor: The predicted polarity
        """
        predicted_polarity = self.classifier(representation)  # Pass the representation through the classifier
        return predicted_polarity
