import gymnasium as gym
from agents import *
from utils import *
from ray import train, tune
from train_utils import train_epoch
import json
from ray.tune.search.optuna import OptunaSearch
from optuna.samplers import TPESampler, CmaEsSampler
from ray.tune.schedulers import ASHAScheduler
import os
import time
import ray

os.environ['RAY_DEDUP_LOGS'] = '0'
if not tr.cuda.is_available(): raise Exception('CUDA unaviable')
ray.init()

def objective(config):
    try:
        device = tr.device('cuda' if tr.cuda.is_available() else 'cpu')
        train_env = gym.make('MountainCarContinuous-v0')
        agent = SACAgent(config,
                        device).to(device)
        # Rewards = 0.
        for epoch in range(int(config['n_epochs'])):
            reward, loss = train_epoch(train_env, agent, epoch, device)
            # Rewards += reward
            train.report({"reward" : reward})

        train_env.close()
    except Exception as e: raise e

    return {"reward" : reward}
    
config = hyperparams_dict("Agent")
search_space = {
    'learning_rate_actor': tune.loguniform(1e-6, 3e-4),
    'learning_rate_critic': tune.loguniform(1e-5, 3e-3),
    'entropy_coeff': tune.uniform(1e-5, 3e-2),
    #'entropy_decay_factor': tune.choice([0.95, 0.99, 1.]),
    # 'num_hidden_layers_actor': tune.choice([2, 3, 4]),
    # 'batch_size' : tune.choice([64, 128, 256, 512]),
    #'grad_clip_actor': tune.uniform(1., 10.0),
    'future_discount_factor': tune.choice([0.9, 0.99, 0.999]),
}
# Overwrite the default config with the search space
for key in search_space.keys():
    config[key] = search_space[key]

# Create an Optuna pruner instance
sampler = TPESampler()
algo = OptunaSearch(sampler=sampler)

scheduler = ASHAScheduler(
    max_t=int(config['n_epochs']),  # The maximum number of training iterations (e.g., epochs)
    grace_period=int(config['n_epochs']//10),    # The number of epochs to run before a trial can be stopped
    reduction_factor=config['reduction_factor'],  # Reduce the number of trials that factor
)

tuner = tune.Tuner(
    tune.with_resources(objective,
                        resources = {'cpu' : 8, 'gpu': 1}),
    tune_config = tune.TuneConfig(
        metric = 'reward',
        mode = 'max',
        search_alg = algo,
        num_samples = config['num_samples'],
        scheduler = scheduler
    ),
    run_config = train.RunConfig(
        verbose=1,
        failure_config=train.FailureConfig(fail_fast=True),
    ),
    param_space = config
)

results = tuner.fit()
fname = 'logs/best_hyperparams.json'
with open(fname, 'w') as f:
    json.dump(results.get_best_result().config, f)

print(results.get_best_result().config)