import configparser
import numpy as np
import ast
import cv2

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


def make_video(env_id, experiment_number, fps=30):
    frames = np.load(f'logs/{env_id}/experiments/{experiment_number}/frames.npy')
    height, width, _ = frames[0].shape
    fourcc = cv2.VideoWriter.fourcc(*'mp4v')
    video = cv2.VideoWriter(f'logs/{env_id}/experiments/{experiment_number}/video.mp4',fourcc, fps, (width, height))
    for frame in frames:
        video.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    video.release()