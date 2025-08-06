"""
Distributed Consensus Gym - Multi-agent RL environment for consensus protocols.

This package provides OpenAI Gym environments for training reinforcement learning
agents to discover and optimize distributed consensus algorithms in the presence
of Byzantine faults, network partitions, and varying system conditions.
"""

from consensus_gym.envs import (
    ByzantineConsensusEnv,
    RaftConsensusEnv,
    PBFTConsensusEnv,
    NetworkPartitionEnv,
)
from consensus_gym.core import ConsensusNode, NetworkSimulator, FaultInjector
from consensus_gym.metrics import ConsensusMetrics, PerformanceAnalyzer

__version__ = "0.1.0"
__author__ = "Juan Flores"

__all__ = [
    "ByzantineConsensusEnv",
    "RaftConsensusEnv", 
    "PBFTConsensusEnv",
    "NetworkPartitionEnv",
    "ConsensusNode",
    "NetworkSimulator",
    "FaultInjector",
    "ConsensusMetrics",
    "PerformanceAnalyzer",
]