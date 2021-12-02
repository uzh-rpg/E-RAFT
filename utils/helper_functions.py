import torch
import os
import smtplib
import json

def move_dict_to_cuda(dictionary_of_tensors, gpu):
    if isinstance(dictionary_of_tensors, dict):
        return {
            key: move_dict_to_cuda(value, gpu)
            for key, value in dictionary_of_tensors.items()
        }
    return dictionary_of_tensors.to(gpu, dtype=torch.float)

def move_list_to_cuda(list_of_dicts, gpu):
    for i in range(len(list_of_dicts)):
        list_of_dicts[i] = move_dict_to_cuda(list_of_dicts[i], gpu)
    return list_of_dicts

def get_values_from_key(input_list, key):
    # Returns all the values with the same key from
    # a list filled with dicts of the same kind
    out = []
    for i in input_list:
        out.append(i[key])
    return out

def create_save_path(subdir, name):
    # Check if sub-folder exists, and create if necessary
    if not os.path.exists(subdir):
        os.mkdir(subdir)
    # Create a new folder (named after the name defined in the config file)
    path = os.path.join(subdir, name)
    # Check if path already exists. if yes -> append a number
    if os.path.exists(path):
        i = 1
        while os.path.exists(path + "_" + str(i)):
            i += 1
        path = path + '_' + str(i)
    os.mkdir(path)
    return path

def get_nth_element_of_all_dict_keys(dict, idx):
    out_dict = {}
    for k in dict.keys():
        d = dict[k][idx]
        if isinstance(d,torch.Tensor):
            out_dict[k]=d.detach().cpu().item()
        else:
            out_dict[k]=d
    return out_dict

def get_number_of_saved_elements(path, template, first=1):
    i = first
    while True:
        if os.path.exists(os.path.join(path,template.format(i))):
            i+=1
        else:
            break
    return range(first, i)

def create_file_path(subdir, name):
    # Check if sub-folder exists, else raise exception
    if not os.path.exists(subdir):
        raise Exception("Path {} does not exist!".format(subdir))
    # Check if file already exists, else create path
    if not os.path.exists(os.path.join(subdir,name)):
        return os.path.join(subdir,name)
    else:
        path = os.path.join(subdir,name)
        prefix,suffix = path.split('.')
        i = 1
        while os.path.exists("{}_{}.{}".format(prefix,i,suffix)):
            i += 1
        return "{}_{}.{}".format(prefix,i,suffix)

def update_dict(dict_old, dict_new):
    # Update all the entries of dict_old with the new values(that have the identical keys) of dict_new
    for k in dict_new.keys():
        if k in dict_old.keys():
            # Replace the entry
            if isinstance(dict_new[k], dict):
                update_dict(dict_old[k], dict_new[k])
            else:
                dict_old[k] = dict_new[k]
    return dict_old
