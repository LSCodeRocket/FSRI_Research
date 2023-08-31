from make_data_loader import *
from neural_network import *
from sys import argv
import matplotlib.pyplot as plt
import numpy as np
import random
from neuralop.models import TFNO1d

model = TFNO1d(n_modes_height=16, hidden_channels=10,
                in_channels=1,
                out_channels=1,
                factorization='tucker',
                implementation='factorized',
                rank=0.05)


learning_rate = 1e-4
epochs = 10

#makes the neural network
#model = NeuralNetwork()
#model = torch.load('model.pth')

#takes the argument the user used to find the file
creep_dict, surface_dict = folder_to_dictionaries(argv[1])

minimum_input = find_maximum_minimum_input(creep_dict)

maximum_input = find_minimum_maximum_input(creep_dict)

#prepares our data for NN training
dataloader = dataloader_tuples(creep_dict, surface_dict)

random.shuffle(dataloader)

#seperates our training data and our test data to ensure our NN works for generalized data and not just the training data
training_dataloader = dataloader[0:(4*len(dataloader) // 5)]
testing_dataloader = dataloader[(4*len(dataloader) //5 ) : -1]

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
    
    random.shuffle(training_dataloader)
    loss.append(test_loop(testing_dataloader, model, loss_fn))
print("Done!")

torch.save(model, 'model.pth')

#plot loss so we can make sure our error is actually decreasing
plt.clf()
plt.plot(np.array(list(range(len(loss)))), np.array(loss))
plt.savefig('output.png')
plt.clf()

random_number = random.randint(0, len(surface_dict))
key = list(surface_dict.keys())[random_number]



positive_input = np.logspace(0,np.log10(30), num=90)
negative_input = []
for value in reversed(positive_input):
    negative_input.append(-value)

full_input = np.array(negative_input + list(positive_input))


positive_x_output = list(dataloader[random_number][1])
negative_x_output = []
for value in reversed(positive_x_output):
    negative_x_output.append(value)

full_output = np.array(negative_x_output + positive_x_output)


positive_x_model = list(model(torch.Tensor(dataloader[random_number][0])).detach().numpy())
negative_x_model = []
for value in reversed(positive_x_model):
    negative_x_model.append(value)

full_model = np.array(negative_x_model + positive_x_model)


plt.plot(full_input, full_output, color="Orange")
plt.plot(full_input, full_model, color="Blue")

plt.savefig("comparison.png")

for param in model.parameters():
    # print("Weights:" + str(param))
    pass

