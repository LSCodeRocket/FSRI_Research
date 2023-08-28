from make_data_loader import *
from neural_network import *
from visualization_neural_network import *
from sys import argv
import matplotlib.pyplot as plt
import numpy as np

learning_rate = 1e-2

epochs = 150

#makes the neural network
model = NeuralNetwork()

#takes the argument the user used to find the file
creep_dict, surface_dict = folder_to_dictionaries(argv[1])

#prepares our data for NN training
dataloader = dataloader_tuples(creep_dict, surface_dict)

#seperates our training data and our test data to ensure our NN works for generalized data and not just the training data
training_dataloader = dataloader[0:(len(dataloader) // 2)]
testing_dataloader = dataloader[(len(dataloader) // 2 ) : -1]

#define loss function
loss_fn = nn.MSELoss()

#define optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

loss = []

#run our NN for our specified epochs
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    #train our NN on all training data
    train_loop(training_dataloader, model, loss_fn, optimizer)
    #store our loss values
    loss.append(test_loop(testing_dataloader, model, loss_fn))
print("Done!")

#plot loss so we can make sure our error is actually decreasing
plt.plot(np.array(list(range(len(loss)))), np.array(loss))
plt.savefig('output.png')