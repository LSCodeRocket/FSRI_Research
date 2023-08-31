import torch
from torch import nn
import numpy as np
from make_data_loader import *

class NeuralOperator(nn.Module):
    def __init__(self):
        super().__init__()
        #make the layers of the NN
        neurons_per_hidden_layer = 100
        hidden_layer_number = 6

        internal_kernels = [nn.Linear(2, 1)]
        print("Neural Network Created.")

    #forward propagation
    def forward(self, x):
        #print("x",x.size())
        logits = self.linear_relu_stack(x)
        #print("logits",logits)
        return logits

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        #make the layers of the NN
        nphl = 100
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_number, nphl),
            nn.SELU(),
            nn.Linear(nphl, nphl),
            nn.SELU(),
            nn.Linear(nphl, nphl),
            nn.SELU(),
            nn.Linear(nphl, nphl),
            nn.SELU(),
            nn.Linear(nphl, nphl),
            nn.SELU(),
            nn.Linear(nphl, nphl),
            nn.SELU(),
            nn.Linear(nphl, nphl),
            nn.SELU(),
            nn.Linear(nphl, 90),
        )
        print("Neural Network Created.")

    #forward propagation
    def forward(self, x):
        #print("x",x.size())
        logits = self.linear_relu_stack(x)
        #print("logits",logits)
        return logits

def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader)
    # Set the model to training mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        X = torch.from_numpy(np.array(X)).float()
        pred = model(X).reshape((90, 1))
        loss = loss_fn(pred, torch.unsqueeze(torch.Tensor(y), 1))


        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch)+1*len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, loss_fn):
    # Set the model to evaluation mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.eval()
    size = len(dataloader)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():
        for X, y in dataloader:

            X = torch.from_numpy(np.array(X).reshape(1, input_number)).float()
            pred = model(X)
            test_loss += loss_fn(torch.Tensor(pred), torch.Tensor(y).reshape(1, 90)).item()

    test_loss /= num_batches
    print(f"Avg loss: {test_loss:>8f} \n")
    return test_loss

