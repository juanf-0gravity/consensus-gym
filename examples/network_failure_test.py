#!/usr/bin/env python3
import gym
import consensus_gym
import numpy as np
from consensus_gym.utils.training import FaultInjector

def test_network_partitions():
    env = gym.make('ConsensusEnv-v0', num_nodes=7)
    fault_injector = FaultInjector()
    
    print("Testing network partitions...")
    
    for test in range(5):
        print(f"\nTest {test + 1}:")
        obs = env.reset()
        
        partition_nodes = np.random.choice(env.num_nodes, size=2, replace=False)
        print(f"Creating partition with nodes: {partition_nodes}")
        
        env.network.create_partition(partition_nodes.tolist())
        
        for step in range(50):
            actions = {i: env.action_space.sample() for i in range(env.num_nodes)}
            obs, rewards, done, info = env.step(actions)
            
            if step == 25:
                print("Healing partition...")
                env.network.heal_partition()
                
            if done:
                print(f"Consensus reached at step {step}")
                break
        else:
            print("No consensus reached")
            
        print(f"Final consensus state: {info.get('consensus_value', 'None')}")

def test_byzantine_faults():
    env = gym.make('ConsensusEnv-v0', num_nodes=7)
    fault_injector = FaultInjector()
    
    print("\nTesting Byzantine faults...")
    
    byzantine_nodes = [0, 2]
    fault_injector.inject_fault(env, 'byzantine', byzantine_nodes)
    
    obs = env.reset()
    for step in range(100):
        actions = {i: env.action_space.sample() for i in range(env.num_nodes)}
        obs, rewards, done, info = env.step(actions)
        
        if done:
            print(f"Consensus with Byzantine nodes reached at step {step}")
            break
    else:
        print("No consensus reached with Byzantine nodes")

if __name__ == "__main__":
    test_network_partitions()
    test_byzantine_faults()