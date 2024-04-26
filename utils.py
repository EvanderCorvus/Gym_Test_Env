import configparser
import numpy as np
import ast

def hyperparams_dict(section):
    config = configparser.ConfigParser()
    config.read('hyperparameters.ini')
    if not config.read('hyperparameters.ini'):
        raise Exception("Could not read config file")
    
    params = config[section]
    typed_params = {}
    for key, value in params.items():
        try:
            # Attempt to evaluate the value to infer type
            typed_params[key] = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            # Fallback to the original string value if evaluation fails
            typed_params[key] = value
    
    return typed_params