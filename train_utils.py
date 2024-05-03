import torch as tr
import gymnasium as gym
import numpy as np
from numpy import random as rnd

def train_epoch(train_env, agent, current_epoch, device, writer = None):
    rewards = 0
    train_env.reset()
    # Enforce Custom Init
    theta_init = rnd.uniform(3*np.pi/4, np.pi)*rnd.choice([-1,1])
    thetadot_init = rnd.uniform(-1.,1)
    train_env.state = np.array([theta_init, thetadot_init])
    state = np.array([np.cos(theta_init), np.sin(theta_init), thetadot_init])

    step = 0
    while True:
        gpu_state = tr.from_numpy(state).to(device).float()
        action = agent.actor(gpu_state).cpu().detach().numpy()
        next_state, reward, terminated, truncated, _ = train_env.step(action)

        agent.replay_buffer.add(state,
                                action,
                                [reward], 
                                next_state)

        loss_actor, loss_critic = agent.update()

        if writer != None:
            writer.add_scalar('loss_actor', loss_actor, (200*current_epoch)+step)
            writer.add_scalar('loss_critic', loss_critic, (200*current_epoch)+step)
            writer.add_scalar('reward', reward, (200*current_epoch)+step)
            writer.add_scalar('action', action, (200*current_epoch)+step)

        rewards += reward
        state = next_state

        if terminated or truncated: break
        step += 1

    loss = loss_actor + loss_critic
    agent.actor_scheduler.step()
    agent.critic_scheduler.step()
    agent.decay_entropy()
    return rewards, loss.item()

def test_loop(agent, device):
    test_env = gym.make('Pendulum-v1', g=9.81, render_mode='human')
    # Test Loop:
    state, _ = test_env.reset()
    for _ in range(800):
        state = tr.from_numpy(state).to(device).float()
        action = agent.act(state)
        env_action = action.cpu().detach().numpy()
        state, _, terminated, truncated, _ = test_env.step(env_action)

        if terminated or truncated:
            state, _ = test_env.reset()            
    test_env.close()