import json
import sys
import glob
import numpy as np
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plot
from progress.bar import Bar

creep_filenames = glob.glob(sys.argv[1] + "/creep_time_depth/*.dat")
surface_filenames = glob.glob(sys.argv[1] + "/surface_r_z/*.dat")



polydeg = int(sys.argv[2])
datapoints = int(sys.argv[3])

creep_dict = {}
bar = Bar("Creep Data", max=len(creep_filenames))
for filename in creep_filenames:
    actual_file = open(filename, 'r')
    actual_file_lines = actual_file.readlines()

    split_actual_file_lines = [actual_file_line.split("-0.0000") for actual_file_line in actual_file_lines]
    float_flipped_lines = [[float(line[1].replace(" ", "")), float(line[0].replace(" ", ""))] for line in split_actual_file_lines][:-2]

    creep_input_data = [data[0] for data in float_flipped_lines]
    creep_output_data = [data[1] for data in float_flipped_lines]

    # Normalized
    creep_input_data = [creep_input_data[n]/max(creep_input_data) for n in range(len(creep_input_data))]

    ones_index = creep_input_data.index(1.0)

    creep_input_data = creep_input_data[0:ones_index+1]
    creep_output_data = creep_output_data[0:ones_index+1]

    creep_input_data = [creep_input_data[n] + 1e-7*n for n in range(len(creep_input_data))]

    # creep_coefficients = np.polyfit(creep_input_data, creep_output_data, deg=polydeg) creep_func = CubicSpline(creep_input_data, creep_output_data) # lambda x: sum([ x**(polydeg-n) * creep_coefficients[n] for n in range(len(creep_coefficients))])
    
    creep_func = CubicSpline(creep_input_data, creep_output_data)
    space = np.linspace(0, 1, num=datapoints)
    creep_datapoint_fitted = [creep_func(space[n]).item() for n in range(len(space))]

    creep_full_fitted = [creep_func(creep_input_data[n]).item() for n in range(len(creep_input_data))]


    filename_formatted = filename.split("/")[-1][:-4].split("_")[-1]
    creep_dict[filename_formatted] = [creep_input_data, creep_output_data, list(space), list(creep_datapoint_fitted), list(creep_full_fitted)] 
    bar.next()
bar.finish()

creep_file = open(sys.argv[1] + "-creep.json", 'w')
json.dump(creep_dict, creep_file)
creep_file.close()

surface_dict = {}
bar = Bar("Surface dict", max = len(surface_filenames))
for filename in surface_filenames:
    actual_file = open(filename, 'r')
    actual_file_lines = actual_file.readlines()

    split_actual_file_lines = [actual_file_line.split(" ") for actual_file_line in actual_file_lines]
    float_lines = [[float(line[0].replace(" ", "")), 100*float(line[1].replace(" ", ""))] for line in split_actual_file_lines]

    surface_input_data = [data[0] for data in float_lines]
    surface_output_data = [data[1] for data in float_lines]

    filename_formatted = filename.split("/")[-1][:-4].split("_")[-1]
    surface_dict[filename_formatted] = [surface_input_data, surface_output_data]
    bar.next()

bar.finish()

surface_file = open(sys.argv[1] + "-surface.json", 'w')
json.dump(surface_dict, surface_file)
surface_file.close()
