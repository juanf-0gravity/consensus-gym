"""Configuration settings for the distributed consensus gym."""

import os
from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class NetworkConfig:
    """Network simulation configuration."""
    latency_min: float = 0.01
    latency_max: float = 0.1
    packet_loss_rate: float = 0.02
    partition_probability: float = 0.05
    max_message_delay: float = 0.5

@dataclass
class ConsensusConfig:
    """Consensus algorithm configuration."""
    num_nodes: int = 5
    byzantine_ratio: float = 0.2
    timeout_duration: float = 1.0
    max_rounds: int = 100
    value_range: tuple = (0, 100)

@dataclass
class TrainingConfig:
    """Training configuration."""
    episodes: int = 1000
    max_steps_per_episode: int = 500
    batch_size: int = 32
    learning_rate: float = 0.001
    replay_buffer_size: int = 10000
    update_frequency: int = 10

@dataclass
class EnvironmentConfig:
    """Main environment configuration."""
    network: NetworkConfig = NetworkConfig()
    consensus: ConsensusConfig = ConsensusConfig()
    training: TrainingConfig = TrainingConfig()
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'EnvironmentConfig':
        """Create config from dictionary."""
        network_config = NetworkConfig(**config_dict.get('network', {}))
        consensus_config = ConsensusConfig(**config_dict.get('consensus', {}))
        training_config = TrainingConfig(**config_dict.get('training', {}))
        
        return cls(
            network=network_config,
            consensus=consensus_config,
            training=training_config
        )

# Default configuration
DEFAULT_CONFIG = EnvironmentConfig()

# Environment-specific configs
TESTING_CONFIG = EnvironmentConfig(
    network=NetworkConfig(packet_loss_rate=0.1, partition_probability=0.2),
    consensus=ConsensusConfig(num_nodes=7, byzantine_ratio=0.3),
    training=TrainingConfig(episodes=100, max_steps_per_episode=200)
)