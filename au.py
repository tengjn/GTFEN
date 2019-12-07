import torch
from collections import OrderedDict

def solve_ori_module_problem(state_dict):

    # create new OrderedDict that does not contain `module.`
    
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] # remove `module.`
        new_state_dict[name] = v
        
    return new_state_dict
    
def solve_module_problem(state_dict):

    # create new OrderedDict that does not contain `module.`
    
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        
        name = k.replace('module.','') # remove `module.`
        new_state_dict[name] = v
        
    return new_state_dict