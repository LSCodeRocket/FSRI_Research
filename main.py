from make_data_loader import *
from neural_network import *
from visualization_neural_network import *
from sys import argv
import matplotlib.pyplot as plt
import numpy as np

learning_rate = 1e-2
epochs = 150

model = NeuralNetwork()

creep_dict, surface_dict = folder_to_dictionaries(argv[1])

dataloader = dataloader_tuples(creep_dict, surface_dict)

training_dataloader = dataloader[0:(len(dataloader) // 2)]
testing_dataloader = dataloader[(len(dataloader) // 2 ) : -1]
loss_fn = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

loss = []
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(training_dataloader, model, loss_fn, optimizer)
    loss.append(test_loop(testing_dataloader, model, loss_fn))
print("Done!")


plt.plot(np.array(list(range(len(loss)))), np.array(loss))
plt.show()