from make_data_loader import *
from neural_network import *
from visualization_neural_network import *
from sys import argv
import matplotlib.pyplot as plt
import numpy as np
import random



learning_rate = 1e-4

epochs = 1000

#makes the neural network
model = NeuralNetwork()

#takes the argument the user used to find the file
creep_dict, surface_dict = folder_to_dictionaries(argv[1])

minimum_input = find_maximum_minimum_input(creep_dict)

maximum_input = find_minimum_maximum_input(creep_dict)

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
plt.clf()
plt.plot(np.array(list(range(len(loss)))), np.array(loss))
plt.savefig('output.png')

key = list(surface_dict.keys())[0]

random_number = random.randint(0, len(surface_dict))

plt.clf()
plt.plot(np.logspace(0,np.log10(30), num=90),np.array(dataloader[random_number][1]), color="Orange")
plt.plot(np.logspace(0,np.log10(30), num=90),model(torch.Tensor(dataloader[random_number][0])).detach().numpy(), color="Blue")

plt.savefig("comparison.png")

for param in model.parameters():
    print("Weights:" + str(param))