import numpy as np
import glob
import json
import os
import scipy as sp


polynomial_degree_approximation = 12
input_number = 120

def import_data_from_filename(filename):
    # Open file in "r"ead mode
    file = open(filename, "r")
    
    # Read all the text within the file
    file_text_raw = file.read()

    # Close the file to prevent some data leaking or something I don't know
    file.close()

    # Split the file_text_raw by new line and then by space
    file_text_lines = file_text_raw.split("\n")
    file_text_variables = [line.replace("                   ","  ").replace("       ","  ").replace("    ","  ").split('  ')[1:] for line in file_text_lines]

    if len(file_text_variables[0]) == 0:
        file_text_variables = [line.split(" ") for line in file_text_lines]

    input_data = [] # Define the input_data list for t or x values
    output_data = [] # Define the output_data list for F or H values

    for line in file_text_variables:
        if len(line) >= 2 and len(line) <= 4:
            input_data.append(float(line[0])) # append the first variable from the text
            output_data.append(float(line[-1])) # append the last variable from the text

    Edot = int(filename.split("_")[-1][4:6])
    N = float(filename.split("_")[-1][7:11])
    Y = float(filename.split("_")[-1][12:-4])
    
    return input_data, output_data, Edot, N, Y #return both numpy arrays for plotting and such (and the variables for the input to the neural network)

def get_all_filenames_in_directory(directory_name):
    # glob.glob() is a function that returns all the folders in a directory with a specific file format
    # In this case, I selected .dat files
    # Then I format the filename so it makes sense to be input directly into the function above.
    files = glob.glob(directory_name + "/*.dat")

    return files 

def put_input_output_data_into_dictionary(directory_name):
    all_files = get_all_filenames_in_directory(directory_name) 
   
    data_dictionary = {}
    for file in all_files:
        data_dictionary[file.split("/")[-1].split("_")[-1]] = import_data_from_filename(file)
    
    return data_dictionary, all_files


def store_data_dictionary_as_json(data_dictionary, json_name):
    json_file = open(json_name, 'w')

    json.dump(data_dictionary, json_file, indent=4)

    json_file.close()

def get_data_from_json(json_name):
    json_file = open(json_name, "r")
    data_dictionary = json.load(json_file)
    json_file.close()


    return data_dictionary


def find_minimum_length_data(data_dictionary):
    return min([len(data[0]) for data in data_dictionary.values()])

def find_maximum_minimum_input(data_dictionary):
    return max([min(data[0]) for data in data_dictionary.values()])
    

def find_minimum_maximum_input(data_dictionary):
    return min([max(data[0]) for data in data_dictionary.values()])

def folder_to_dictionaries(directory_name):
    print("Taking the folder named " + directory_name + " to get the dictionary.")

    if os.path.isfile(directory_name + "-surface_r_z.json"):
        surface_dict = get_data_from_json(directory_name + "-surface_r_z.json")
        print("Got Json file for surface")
    else:
        surface_dict, _ = put_input_output_data_into_dictionary(directory_name + "/surface_r_z")
        store_data_dictionary_as_json(surface_dict, directory_name + "-surface_r_z.json")
        print("made json file for surface")
    if os.path.isfile(directory_name + "-creep_time_depth.json"):
        print("got JSON File for creep ")
        creep_dict = get_data_from_json(directory_name + "-creep_time_depth.json")
    else:
        creep_dict, _ = put_input_output_data_into_dictionary(directory_name + "/creep_time_depth")
        store_data_dictionary_as_json(creep_dict, directory_name + "-creep_time_depth.json")
        print("made json creep file")
    
    return creep_dict, surface_dict

def dataloader_tuples(creep_data_dict, surface_data_dict):
    print("hard part begins")
    output_data = []
    for curve_key in creep_data_dict.keys():
        surface = surface_data_dict[curve_key]
        creep = creep_data_dict[curve_key]

        creep_fitted_coefficients = np.polyfit(creep[0], creep[1], polynomial_degree_approximation)

        creep_func = lambda t: sum([ t**(polynomial_degree_approximation-n) * creep_fitted_coefficients[n] for n in range(polynomial_degree_approximation) ])


        input_max = max(creep[0])
        input_min = min(creep[0])

        new_creep = [creep_func(x) for x in np.linspace(input_min, input_max, num=input_number)]
        output_data.append((new_creep, surface[1]))

    return output_data
