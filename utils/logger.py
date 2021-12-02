import json
import os
import numpy
import shutil

class Logger:
    # Logger of the Training/Testing Process
    def __init__(self, save_path, custom_name='log.txt'):
        self.toWrite = {}
        self.signalization = "========================================"
        self.path = os.path.join(save_path,custom_name)

    def initialize_file(self, mode):
        # Mode : "Training" or "Testing"
        with open(self.path, 'a') as file:
            file.write(self.signalization + " " + mode + " " + self.signalization + "\n")

    def write_as_list(self, dict_to_write, overwrite=False):
        if overwrite:
            if os.path.exists(self.path):
                os.remove(self.path)
        with open(self.path, 'a') as file:
            for entry in dict_to_write.keys():
                file.write(entry+"="+json.dumps(dict_to_write[entry])+"\n")

    def write_dict(self, dict_to_write, array_names=None, overwrite=False, as_list=False):
        if overwrite:
            open_type = 'w'
        else:
            open_type = 'a'
        dict_to_write = self.check_for_arrays(dict_to_write, array_names)
        if as_list:
            self.write_as_list(dict_to_write, overwrite)
        else:
            with open(self.path, open_type) as file:
                #if "epoch" in dict_to_write:
                 #   file.write("Epoch")
                file.write(json.dumps(dict_to_write) + "\n")

    def write_line(self,line, verbose=False):
        with open(self.path, 'a') as file:
            file.write(line + "\n")
        if verbose:
            print(line)

    def arrays_to_dicts(self, list_of_arrays, array_name, entry_name):
        list_of_arrays = numpy.array(list_of_arrays).T
        out = {}
        for i in range(list_of_arrays.shape[0]):
            out[array_name+'_'+entry_name[i]] = list(list_of_arrays[i])
        return out


    def check_for_arrays(self, dict_to_write, array_names):
        if array_names is not None:
            names = []
            for n in range(len(array_names)):
                if hasattr(array_names[n], 'name'):
                    names.append(array_names[n].name)
                elif hasattr(array_names[n],'__name__'):
                    names.append(array_names[n].__name__)
                elif hasattr(array_names[n],'__class__'):
                    names.append(array_names[n].__class__.__name__)
                else:
                    names.append(array_names[n])

        keys = dict_to_write.keys()
        out = {}
        for entry in keys:
            if hasattr(dict_to_write[entry], '__len__') and len(dict_to_write[entry])>0:
                if isinstance(dict_to_write[entry][0], numpy.ndarray) or isinstance(dict_to_write[entry][0], list):
                    out.update(self.arrays_to_dicts(dict_to_write[entry], entry, names))
                else:
                    out.update({entry:dict_to_write[entry]})
            else:
                out.update({entry: dict_to_write[entry]})
        return out
