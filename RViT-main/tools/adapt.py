import torch
import torch.nn as nn

# Define the FeedForward class
class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        # Define a simple feed-forward network with GELU activation and dropout
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # Pass the input through the feed-forward network
        return self.net(x)


# Define the FC_class that uses the FeedForward network
class FC_class(nn.Module):
    def __init__(self, input_dim, class_num):
        super(FC_class, self).__init__()
        # Use a feed-forward network to increase dimensionality before classification
        self.mlp = FeedForward(input_dim, input_dim * 4, 0.1)
        # Final classification layer that maps to the number of classes
        self.class_ = nn.Linear(input_dim, class_num)

    def forward(self, embedding):
        # Apply the feed-forward network and then classify
        mlp_embedding = self.mlp(embedding)
        return self.class_(mlp_embedding), mlp_embedding
