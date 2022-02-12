import numpy as np
import gym
from typing import Dict, List, Any, Optional
import torch
import torch.nn as nn
from collections import deque

class MultiAgentTrainer:
    def __init__(self, env, agents, buffer_size=10000):
        self.env = env
        self.agents = agents
        self.replay_buffer = deque(maxlen=buffer_size)
        self.episode_rewards = []
        
    def train_episode(self, max_steps=1000):
        obs = self.env.reset()
        episode_reward = 0
        
        for step in range(max_steps):
            actions = {}
            for agent_id, agent in self.agents.items():
                actions[agent_id] = agent.act(obs[agent_id])
                
            next_obs, rewards, done, info = self.env.step(actions)
            
            for agent_id in self.agents:
                transition = {
                    'obs': obs[agent_id],
                    'action': actions[agent_id],
                    'reward': rewards[agent_id],
                    'next_obs': next_obs[agent_id],
                    'done': done
                }
                self.replay_buffer.append(transition)
                
            obs = next_obs
            episode_reward += sum(rewards.values())
            
            if done:
                break
                
        self.episode_rewards.append(episode_reward)
        return episode_reward
        
    def update_agents(self, batch_size=32):
        if len(self.replay_buffer) < batch_size:
            return
            
        batch = np.random.choice(len(self.replay_buffer), batch_size, replace=False)
        
        for agent in self.agents.values():
            if hasattr(agent, 'update'):
                agent.update([self.replay_buffer[i] for i in batch])

class FaultInjector:
    def __init__(self):
        self.fault_types = ['byzantine', 'crash', 'slow']
        
    def inject_fault(self, env, fault_type, node_ids, duration=None):
        if fault_type == 'byzantine':
            env.make_byzantine(node_ids)
        elif fault_type == 'crash':
            env.crash_nodes(node_ids)
        elif fault_type == 'slow':
            env.slow_nodes(node_ids, factor=0.5)
            
    def random_fault(self, env, probability=0.1):
        if np.random.random() < probability:
            fault_type = np.random.choice(self.fault_types)
            num_nodes = env.num_nodes
            node_id = np.random.randint(0, num_nodes)
            self.inject_fault(env, fault_type, [node_id])