import numpy as np
import glob
import json
import os

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

    input_data = [] # Define the input_data list for t or x values
    output_data = [] # Define the output_data list for F or H values

    for line in file_text_variables:
        if len(line) >= 2 and len(line) <= 4:
            input_data.append(float(line[0])) # append the first variable from the text
            output_data.append(float(line[-1])) # append the last variable from the text

    Edot = int(filename[4:7])
    N = float(filename[7:11])
    Y = float(filename[12:-1])
    
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
        data_dictionary[file] = import_data_from_filename(file)
    
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


def dataloader_tuples(creep_data_dict, surface_data_dict):
    for curve_key in creep_data_dict.keys():
        surface = surface_data_dict[curve_key]
        creep = creep_data_dict[curve_key]