import numpy as np
import glob
import json
import os
import scipy as sp


polynomial_degree_approximation = 12
input_number = 120

def get_data_from_json(json_name):
    #open file in read mode
    json_file = open(json_name, "r")

    #extract data from file
    data_dictionary = json.load(json_file)

    #close file to prevent data-leaking
    json_file.close()


    return data_dictionary


def find_minimum_length_data(data_dictionary):
    #unused
    return min([len(data[0]) for data in data_dictionary.values()])

def find_maximum_minimum_input(data_dictionary):
    #finds the largest minimum of all graphs
    return max([min(data[0]) for data in data_dictionary.values()])
    

def find_minimum_maximum_input(data_dictionary):
    #finds the smallest maximum of all graphs
    return min([max(data[0]) for data in data_dictionary.values()])

def folder_to_dictionaries(directory_name):
    #print statement for checkpoints
    print("Taking the folder named " + directory_name + " to get the dictionary.")
    surface_dict = get_data_from_json(directory_name + "-surface.json")
    creep_dict = get_data_from_json(directory_name + "-creep.json")
    
    return creep_dict, surface_dict

def dataloader_tuples(creep_data_dict, surface_data_dict):
    #checkpoint
    print("hard part begins")


    output_data = []
    for curve_key in creep_data_dict.keys():
        surface = surface_data_dict[curve_key]
        creep = creep_data_dict[curve_key]

        output_data.append((creep[3], surface[1]))

    return output_data
