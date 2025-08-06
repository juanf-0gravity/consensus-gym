from .consensus_env import ConsensusEnv

# register with gym
from gym.envs.registration import register

register(
    id='Consensus-v0',
    entry_point='consensus_gym.envs:ConsensusEnv',
    max_episode_steps=1000,
)

__all__ = ['ConsensusEnv']