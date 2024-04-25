import gymnasium as gym
from agents import *
from utils import *
from ray import train, tune
from gym_train_utils import train_epoch
import json
from ray.tune.search.optuna import OptunaSearch
from optuna.samplers import TPESampler, CmaEsSampler
from ray.tune.schedulers import ASHAScheduler
import time

if not tr.cuda.is_available(): raise Exception('CUDA unaviable')

def objective(config):
    device = tr.device('cuda' if tr.cuda.is_available() else 'cpu')
    train_env = gym.make('Pendulum-v1', g=9.81)
    agent = SACAgent(config,
                     device).to(device)
    
    Rewards = 0.
    for epoch in range(int(config['n_epochs'])):
        reward, loss = train_epoch(train_env, agent, epoch, device)
        if np.isnan(loss): print("Loss is NaN")
        Rewards += reward
        train.report({"loss" : loss})

    train_env.close()
    return {"loss" : loss}
    
config = hyperparams_dict("Agent")
search_space = {
    'learning_rate_actor': tune.loguniform(1e-4, 1e-2),  #tune.grid_search(np.linspace(1e-5, 3e-3, 5)),
    'entropy_coeff': tune.uniform(0.01, 0.1),
    'hidden_dims_actor': tune.choice([[64, 64, 64], [512, 512, 512], [1024, 1024, 1024]]),
    'future_discount_factor' : tune.choice([0.9, 0.99, 0.999])
}
# Overwrite the default config with the search space
for key in search_space.keys():
    config[key] = search_space[key]

# Create an Optuna pruner instance
sampler = TPESampler()
algo = OptunaSearch(sampler = sampler)

scheduler = ASHAScheduler(
    max_t=int(config['n_epochs']),  # The maximum number of training iterations (e.g., epochs)
    grace_period=10,    # The number of epochs to run before a trial can be stopped
    reduction_factor=2,  # Reduce the number of trials by a factor of 2 each round
)

tuner = tune.Tuner(
    tune.with_resources(objective,
                        resources = {'cpu' : 8, 'gpu': 1}),
    #objective,
    tune_config = tune.TuneConfig(
        metric = 'loss',
        mode = 'min',
        search_alg = algo,
        num_samples = 2,
        scheduler = scheduler
    ),
    run_config = train.RunConfig(
        verbose=1
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

print(results.get_best_result(metric="loss", mode="min").config)