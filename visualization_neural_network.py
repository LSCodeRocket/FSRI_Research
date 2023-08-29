import numpy as np
import torch
import matplotlib.pyplot as plt
from make_data_loader import *
import random
import sys
from progress.bar import Bar

creep_dict, surface_dict = folder_to_dictionaries(sys.argv[1])

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