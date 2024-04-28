import gymnasium as gym
from agents import *
from utils import *
from ray import train, tune
from train_utils import train_epoch
import json
from ray.tune.search.optuna import OptunaSearch
from optuna.samplers import TPESampler, CmaEsSampler
from ray.tune.schedulers import ASHAScheduler
import time
import ray

if not tr.cuda.is_available(): raise Exception('CUDA unaviable')
# ray.init(object_store_memory=int(16e9)) # Limit Memory to 16GB

def objective(config):
    try:
        device = tr.device('cuda' if tr.cuda.is_available() else 'cpu')
        train_env = gym.make('Pendulum-v1', g=9.81)
        agent = SACAgent(config,
                        device).to(device)
        Rewards = 0.
        for epoch in range(int(config['n_epochs'])):
            reward, loss = train_epoch(train_env, agent, epoch, device, decay_entropy=True)
            Rewards += reward
            train.report({"reward" : reward})

        train_env.close()
    except Exception as e: raise e

    return {"reward" : reward}
    
config = hyperparams_dict("Agent")
search_space = {
    'learning_rate_actor': tune.loguniform(1e-5, 1e-3),  #tune.grid_search(np.linspace(1e-5, 3e-3, 5)),
    'entropy_coeff': tune.uniform(1e-4, 1e-2),
    # 'hidden_dims_actor': tune.choice([256, 512, 1024]),
    # 'num_hidden_layers_actor': tune.choice([2, 3, 4]),
}
# Overwrite the default config with the search space
for key in search_space.keys():
    config[key] = search_space[key]

# Create an Optuna pruner instance
sampler = TPESampler()
algo = OptunaSearch(sampler = sampler)

scheduler = ASHAScheduler(
    max_t=int(config['n_epochs']),  # The maximum number of training iterations (e.g., epochs)
    grace_period=int(config['n_epochs']//10),    # The number of epochs to run before a trial can be stopped
    reduction_factor=config['reduction_factor'],  # Reduce the number of trials that factor
)

tuner = tune.Tuner(
    tune.with_resources(objective,
                        resources = {'cpu' : 8, 'gpu': 1}),
    #objective,
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

start = time.time()
results = tuner.fit()
end = time.time()
print(f"Fit Time: {np.round((end-start)/60,1)} Minutes")
fname = 'logs/best_hyperparams.json'
with open(fname, 'w') as f:
    json.dump(results.get_best_result().config, f)

print(results.get_best_result().config)