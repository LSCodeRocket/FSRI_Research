import numpy as np
import torch
import matplotlib.pyplot as plt
from make_data_loader import *
import random
import sys
from progress.bar import Bar

network_model_filename = sys.argv[3] 
operator_model_filename = sys.argv[4] 

creep_dict, surface_dict = folder_to_dictionaries(sys.argv[1])


def display_comparison_graph():
    model = torch.load(network_model_filename)
    operator = torch.load(operator_model_filename)

    keys = list(surface_dict.keys())

    bar = Bar("Making Curves", max=int(sys.argv[2]))

    input_min = find_maximum_minimum_input(creep_dict)
    input_max = find_minimum_maximum_input(creep_dict)

    max_parameter = int(sys.argv[2])


    for i in range(int(sys.argv[2])):

        random_curve_id = random.randint(0, len(keys)-1)

        creep = creep_dict[keys[random_curve_id]]
        surface = surface_dict[keys[0]]

        positive_input = list(surface[0])
        negative_input = []
        for value in reversed(positive_input):
            negative_input.append(-value)

        full_input = np.array(negative_input + list(positive_input))

        x = np.linspace(0, 1, num=len(creep[3]))
        creep[3] = list(x+(i/max_parameter)*np.sin((10*i/max_parameter)*x+i/max_parameter))

        positive_x_network_model = list(model(torch.Tensor(creep[3])).detach().numpy())
        negative_x_network_model = []
        for value in reversed(positive_x_network_model):
            negative_x_network_model.append(value)
        full_model_network = np.array(negative_x_network_model + positive_x_network_model)

        positive_x_operator_model = list(operator(torch.Tensor(creep[3])).detach().numpy())
        negative_x_operator_model = []
        for value in reversed(positive_x_operator_model):
            negative_x_operator_model.append(value)

        full_model_operator = np.array(negative_x_operator_model + positive_x_operator_model)

        plt.cla()
        plt.figure(figsize=(10,6))
        fig, axes = plt.subplots(1, 3)
        fig.set_figwidth(15)
        # axes[1].title = (f"Surface {keys[random_curve_id][:-4]}")
        # axes[1].plot(full_input, (1/100)*full_output, color='orange')
        axes[1].plot(full_input, (1/100)*full_model_network, color="seagreen")
        axes[1].plot(full_input, (1/100)*full_model_operator, color="brown")
        axes[2].plot(full_input, (1/100)*(np.array(full_model_network)-np.array(full_model_operator)))

        positive_input = list(creep[0])
        negative_input = list(-np.array(positive_input))
        negative_input.reverse()

        full_input = np.array(negative_input + list(positive_input))

        positive_x_output = list(creep[1])
        negative_x_output = positive_x_output
        negative_x_output.reverse()

        full_output = np.array(negative_x_output + positive_x_output)


        # axes[0].title = (f"Creep {keys[random_curve_id][:-4]}")
        axes[0].plot( np.array(creep[2]), np.array(creep[3]), color='red')
        axes[0].plot("Example New Prediction")
        fig.savefig(f"special_testing/{keys[random_curve_id][:-4]}.png")
        plt.clf()
        bar.next()
        plt.close()
    bar.finish()

display_comparison_graph()