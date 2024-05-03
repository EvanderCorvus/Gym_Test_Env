from train_utils import train_epoch, test_loop
from tqdm import tqdm
import gymnasium as gym
import os
import json
from utils import *
from agents import *

from torch.utils.tensorboard.writer import SummaryWriter
if not tr.cuda.is_available(): raise Exception('CUDA unaviable')
device = tr.device('cuda' if tr.cuda.is_available() else 'cpu')

with open('logs/best_hyperparams.json', 'r') as f:
    config = json.load(f)

# config = hyperparams_dict("Agent")
train_env = gym.make('Pendulum-v1', g=9.81)


agent = SACAgent(config,
                 device).to(device)

experiment_number = len(os.listdir('logs/experiments'))+1
os.makedirs(f'logs/experiments/{experiment_number}')
writer = SummaryWriter(f'logs/experiments/{experiment_number}')
for key, value in config.items():
        writer.add_text(key, str(value))
# writer.add_hparams(config, {})


for epoch in tqdm(range(int(config['n_epochs']))):
    reward, loss = train_epoch(train_env, agent, epoch, device, writer)
    if np.isnan(loss):
        raise ValueError("Loss is NaN")
writer.close()

test_loop(agent, device)
