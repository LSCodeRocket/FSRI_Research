import numpy as np
import torch
import matplotlib.pyplot as plt
from make_data_loader import *
import random
import sys
from progress.bar import Bar

creep_dict, surface_dict = folder_to_dictionaries(sys.argv[1])

def display_creep_curves():

    keys = list(creep_dict.keys())

    bar = Bar("Making Curves", max=int(sys.argv[2]))

    for i in range(int(sys.argv[2])):

        random_curve_id = random.randint(0, len(keys))
        creep_curve = creep_dict[keys[random_curve_id]]

        input_min = min(creep_curve[0])
        input_max = max(creep_curve[0])

        positive_input = np.linspace(input_min, input_max, num=len(creep_curve[0]))
        negative_input = list(-positive_input)
        negative_input.reverse()

        full_input = np.array(negative_input + list(positive_input))

        positive_x_output = list(creep_curve[1])
        negative_x_output = positive_x_output
        negative_x_output.reverse()

        full_output = np.array(negative_x_output + positive_x_output)

        plt.clf()
        plt.title(f"{keys[random_curve_id]}")
        plt.plot(positive_input, positive_x_output)
        plt.savefig(f"creep_curves/{keys[random_curve_id][:-4]}.png")
        bar.next()
    bar.finish()



def display_surface_curves():
    model = torch.load("model_50nphl_4hl.pth")

    keys = list(surface_dict.keys())

    bar = Bar("Making Curves", max=int(sys.argv[2]))

    input_min = find_maximum_minimum_input(creep_dict)
    input_max = find_minimum_maximum_input(creep_dict)

    for i in range(int(sys.argv[2])):

        random_curve_id = random.randint(0, len(keys))
        surface_curve = surface_dict[keys[random_curve_id]]

        positive_input = list(np.logspace(0, np.log10(30), num=90))
        negative_input = []
        for value in reversed(positive_input):
            negative_input.append(-value)

        full_input = np.array(negative_input + list(positive_input))

        positive_x_output = list(surface_curve[1])
        negative_x_output = []
        for value in reversed(positive_x_output):
            negative_x_output.append(value)

        full_output = np.array(negative_x_output + positive_x_output)

        creep_input = creep_dict[keys[random_curve_id]][0]
        creep_input_curve = creep_dict[keys[random_curve_id]][1]
        
        creep_fitted_coefficients = np.polyfit(creep_input, creep_input_curve, deg=polynomial_degree_approximation)

        #make function for fitted curve
        creep_func = lambda t: sum([ t**(polynomial_degree_approximation-n) * creep_fitted_coefficients[n] for n in range(polynomial_degree_approximation) ])

        #evaluate new curve at specific points across all new curves
        new_creep = [creep_func(x) for x in np.linspace(input_min, input_max, num=input_number)]


        positive_x_model = list(model(torch.Tensor(new_creep)).detach().numpy())
        negative_x_model = []
        for value in reversed(positive_x_model):
            negative_x_model.append(value)

        full_model = np.array(negative_x_model + positive_x_model)

        plt.clf()
        plt.title(f"{keys[random_curve_id][:-4]}")
        plt.plot(full_input, full_output)
        #plt.plot(full_input, full_model)
        plt.savefig(f"pure_surface_curves/{keys[random_curve_id][:-4]}.png")
        bar.next()
    bar.finish()



def display_surface_creep_curves():
    model = torch.load("model_50nphl_4hl.pth")

    keys = list(surface_dict.keys())

    bar = Bar("Making Curves", max=int(sys.argv[2]))

    input_min = find_maximum_minimum_input(creep_dict)
    input_max = find_minimum_maximum_input(creep_dict)

    for i in range(int(sys.argv[2])):

        random_curve_id = random.randint(0, len(keys))
        surface_curve = surface_dict[keys[random_curve_id]]

        positive_input = list(np.logspace(0, np.log10(30), num=90))
        negative_input = []
        for value in reversed(positive_input):
            negative_input.append(-value)

        full_input = np.array(negative_input + list(positive_input))

        positive_x_output = list(surface_curve[1])
        negative_x_output = []
        for value in reversed(positive_x_output):
            negative_x_output.append(value)

        full_output = np.array(negative_x_output + positive_x_output)

        creep_input = creep_dict[keys[random_curve_id]][0]
        creep_input_curve = creep_dict[keys[random_curve_id]][1]
        
        creep_fitted_coefficients = np.polyfit(creep_input, creep_input_curve, deg=polynomial_degree_approximation)

        #make function for fitted curve
        creep_func = lambda t: sum([ t**(polynomial_degree_approximation-n) * creep_fitted_coefficients[n] for n in range(polynomial_degree_approximation) ])

        #evaluate new curve at specific points across all new curves
        new_creep = [creep_func(x) for x in np.linspace(input_min, input_max, num=input_number)]


        positive_x_model = list(model(torch.Tensor(new_creep)).detach().numpy())
        negative_x_model = []
        for value in reversed(positive_x_model):
            negative_x_model.append(value)

        full_model = np.array(negative_x_model + positive_x_model)

        plt.cla()
        plt.figure(figsize=(10,6))
        fig, axes = plt.subplots(1, 2)
        fig.set_figwidth(15)
        # axes[1].title = (f"Surface {keys[random_curve_id][:-4]}")
        axes[1].plot(full_input, full_output, color='orange')
        axes[1].plot(full_input, full_model, color="seagreen")

        creep_curve = creep_dict[keys[random_curve_id]]

        input_min = min(creep_curve[0])
        input_max = max(creep_curve[0])

        positive_input = np.linspace(input_min, input_max, num=len(creep_curve[0]))
        negative_input = list(-positive_input)
        negative_input.reverse()

        full_input = np.array(negative_input + list(positive_input))

        positive_x_output = list(creep_curve[1])
        negative_x_output = positive_x_output
        negative_x_output.reverse()

        full_output = np.array(negative_x_output + positive_x_output)


        # axes[0].title = (f"Creep {keys[random_curve_id][:-4]}")
        axes[0].plot(positive_input, positive_x_output, color='blue')
        fig.savefig(f"combination_curves/{keys[random_curve_id][:-4]}.png")
        bar.next()
        plt.close()
    bar.finish()

display_surface_creep_curves()