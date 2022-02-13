"""Distributed Consensus Gym - Multi-agent RL environment for consensus algorithms."""

from gym.envs.registration import register
from consensus_gym.envs.consensus_env import ConsensusEnv
from consensus_gym.envs.byzantine_env import ByzantineConsensusEnv
from consensus_gym.core.network import NetworkSimulator
from consensus_gym.utils.training import MultiAgentTrainer, FaultInjector
from consensus_gym.utils.visualization import ConsensusVisualizer
from consensus_gym.utils.metrics import MetricsCollector

register(
    id='ConsensusEnv-v0',
    entry_point='consensus_gym.envs:ConsensusEnv',
    max_episode_steps=1000,
)

register(
    id='ByzantineConsensusEnv-v0',
    entry_point='consensus_gym.envs:ByzantineConsensusEnv',
    max_episode_steps=1000,
)

__version__ = "0.1.0"