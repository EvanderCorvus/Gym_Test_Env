import numpy as np
import gymnasium as gym
from gymnasium import spaces
import random as rand

class Environment(gym.Env):
    observation_space = None
    def __init__(self, config):
        super(Environment, self).__init__()
        self.n_agents = config["n_agents"]
        self.size = config["size"]
        self.vision_range = config["vision_range"]
        self.dt = config["dt"]
        self.characteristic_length = config["characteristic_length"]
        self.sigma = config["sigma"]
        self.epsilon = config["epsilon"]

        #self.seed(config["seed"])

        if self.observation_space == None: 
            Environment.configure_spaces(self.size, self.n_agents)#, self.seed_val)
        
        self.states = None

    def reset(self):
        return self.observation_space.sample()
    
    def step(self, action_dict):
        next_states, rewards = {}, {}
        forces = {f"agent_{i}": np.zeros(2, dtype=np.float64) for i in range(self.n_agents)}

        for i in range(self.n_agents):
            pos_i = self.states[f"agent_{i}"]#[0]
            #theta = self.states[f"agent_{i}"][1]
            action = action_dict[f"agent_{i}"].cpu().detach().numpy()
            
            noise = np.random.normal(0, 1)
            next_theta = action + np.sqrt(self.dt)*self.characteristic_length*noise

            # for j in range(i+1, self.n_agents):
            #     pos_j = self.states[f"agent_{j}"]
            #     force = self._wca_force(pos_i, pos_j)
            #     forces[f"agent_{i}"] = forces[f"agent_{i}"] + force
            #     forces[f"agent_{j}"] = forces[f"agent_{i}"] - force
            
            e = np.array([np.cos(next_theta)[0], np.sin(next_theta)[0]]).T
            v_f = forces[f"agent_{i}"]*self.dt
            v = e + v_f
            next_pos = pos_i + v*self.dt

            
            if not self.observation_space[f'agent_{i}'].contains(next_pos):
                next_pos = np.clip(next_pos, 0., self.size)

            next_states[f"agent_{i}"] = next_pos #(next_pos, next_theta)


        #rewards = self._get_reward(next_states)
        rewards = self._dummy_reward(next_states)

        # Sanity Check
        if not self.observation_space.contains(next_states): raise ValueError("Invalid states", next_states)
        
        return next_states, rewards #, False, False, {} # (Gym API compatibility)


    def _get_reward(self, states):
        rewards = {f"agent_{i}": np.zeros((1,), dtype = np.float64) for i in range(self.n_agents)}
        theta_group = np.array([states[f"agent_{i}"][1] for i in range(self.n_agents)]).mean()

        for i in range(self.n_agents):
            pos_i = states[f"agent_{i}"][0]
            local_moment_of_inertia = np.zeros((2,2), dtype = np.float64)

            for j in range(0, self.n_agents):
                if i==j: continue
                pos_j = states[f"agent_{j}"][0]
                r = np.linalg.norm(pos_j - pos_i)
                if r < self.vision_range:
                    relative_pos = pos_j - pos_i
                    for k in range(2):
                        for l in range(2):
                            local_moment_of_inertia[k][l] += r**2 - relative_pos[k]*relative_pos[l]

            # Change reference frame of local moment of inertia and calculate EVs
            theta_rot = theta_group - states[f"agent_{i}"][1]
            rotation_matrix = np.array([[np.cos(theta_rot[0]), -np.sin(theta_rot[0])],
                                        [np.sin(theta_rot[0]), np.cos(theta_rot[0])]])
            
            local_moment_of_inertia = np.dot(local_moment_of_inertia, rotation_matrix)
            
            eigenvalues = np.linalg.eigvals(local_moment_of_inertia)

            # EV1 - EV0 rewards aligment perpendicualr to axis, EV0 - EV1 rewards aligment parallel to axis
            rewards[f"agent_{i}"] += np.array(eigenvalues[1] - eigenvalues[0])

            # check if agent at the edge of the box
            if (pos_i[0] == 0. or pos_i[0] == self.size) or (pos_i[1] == 0. or pos_i[1] == self.size):
                rewards[f"agent_{i}"] -= 1.0

        return rewards
    
    # Dummy Reward that rewards being in the first quadrant
    def _dummy_reward(self, states):
        rewards = {}
        for i in range(self.n_agents):
            pos_i = states[f"agent_{i}"][0]
            r = np.linalg.norm(pos_i - np.array([0.5, 0.5]))
            if r < 0.1:
                rewards[f"agent_{i}"] = np.array([1.0])
            else:
                rewards[f"agent_{i}"] = np.array([1*np.exp(-5*r)])
            if np.isin(pos_i, [0., self.size]).any():
                rewards[f"agent_{i}"] -= np.array([.5])

        return rewards
    
    def _dummy_dummy_reward(self, states):
        rewards = {}
        for i in range(self.n_agents):
            y = states[f"agent_{i}"][0][1]
            rewards[f"agent_{i}"] = np.array([np.exp(-1*y)])
        
        return rewards

    @classmethod
    def configure_spaces(cls, size, n_agents):
        cls.action_space = spaces.Dict({f"agent_{i}": spaces.Box(-np.pi, np.pi, shape=(1,1), dtype=float) for i in range(n_agents)})
        cls.observation_space = spaces.Dict({f"agent_{i}": spaces.Box(0, size, shape=(1,2), dtype=float) for i in range(n_agents)})

    def seed(self, seed=None):
        self.seed_val = seed
        rand.seed(seed)
        np.random.seed(seed)
