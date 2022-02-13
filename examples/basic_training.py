#!/usr/bin/env python3
import gym
import consensus_gym
from consensus_gym.utils.training import MultiAgentTrainer, FaultInjector

class SimpleAgent:
    def __init__(self, action_space):
        self.action_space = action_space
        
    def act(self, obs):
        return self.action_space.sample()
        
    def update(self, batch):
        pass

def main():
    env = gym.make('ConsensusEnv-v0', num_nodes=5)
    
    agents = {}
    for i in range(env.num_nodes):
        agents[i] = SimpleAgent(env.action_space)
        
    trainer = MultiAgentTrainer(env, agents)
    fault_injector = FaultInjector()
    
    for episode in range(100):
        reward = trainer.train_episode()
        
        if episode % 10 == 0:
            fault_injector.random_fault(env, probability=0.2)
            
        if episode % 20 == 0:
            print(f"Episode {episode}, Reward: {reward:.2f}")
            
        trainer.update_agents()

if __name__ == "__main__":
    main()