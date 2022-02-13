# Distributed Consensus Gym

A reinforcement learning environment for studying distributed consensus algorithms with fault tolerance.

## Overview

This gym environment simulates a distributed system where agents must reach consensus while handling various failure modes including Byzantine faults, network partitions, and node crashes.

## Features

- **Multi-agent RL Environment**: Compatible with OpenAI Gym
- **Network Simulation**: Realistic network delays, packet loss, and partitions
- **Fault Injection**: Byzantine nodes, crashes, and slow nodes
- **Consensus Algorithms**: Pluggable consensus implementations
- **Training Utilities**: Multi-agent training with fault scenarios

## Quick Start

```bash
pip install -r requirements.txt
python examples/basic_training.py
```

## Usage Examples

### Basic Training
```python
import gym
import consensus_gym

env = gym.make('ConsensusEnv-v0', num_nodes=5)
obs = env.reset()

for step in range(1000):
    actions = {i: env.action_space.sample() for i in range(env.num_nodes)}
    obs, rewards, done, info = env.step(actions)
    if done:
        print(f"Consensus reached: {info['consensus_value']}")
        break
```

### Network Fault Testing
```python
from consensus_gym.utils.training import FaultInjector

fault_injector = FaultInjector()
fault_injector.inject_fault(env, 'byzantine', [0, 1])
env.network.create_partition([2, 3])
```

## Environment Details

### Observation Space
- Node state vectors
- Message queues
- Network topology

### Action Space  
- Consensus proposals
- Message broadcasts
- Vote decisions

### Reward Structure
- Consensus achievement: +10
- Time penalty: -0.1 per step
- Byzantine behavior: -5

## Configuration

See `config.py` for environment parameters:
- Network latency and loss rates
- Byzantine fault ratios
- Training hyperparameters