import torch as tr
import gymnasium as gym

def train_epoch(train_env, agent, current_epoch, device, writer = None):
    rewards = 0
    state, _ = train_env.reset()
    step = 0
    while True:
        state = tr.from_numpy(state).to(device).float()
        action = agent.actor(state)
        env_action = action.cpu().detach().numpy()
        next_state, reward, terminated, truncated, _ = train_env.step(env_action)

        agent.replay_buffer.add(state.unsqueeze(0).float(), action.unsqueeze(0).float().detach(),
                                tr.tensor([reward]).float().to(device), 
                                tr.from_numpy(next_state).unsqueeze(0).float().to(device))

        loss_actor, loss_critic = agent.update(1)

        if writer != None:
            writer.add_scalar('loss_actor', loss_actor, (200*current_epoch)+step)
            writer.add_scalar('loss_critic', loss_critic, (200*current_epoch)+step)
            writer.add_scalar('reward', reward, (200*current_epoch)+step)
            writer.add_scalar('action', env_action, (200*current_epoch)+step)

        rewards += reward
        state = next_state

        if terminated or truncated: break
        step += 1

    loss = loss_actor + loss_critic
    agent.actor_scheduler.step(loss_actor)
    agent.critic_scheduler.step(loss_critic)
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