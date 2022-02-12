"""Byzantine fault-tolerant consensus environment."""

import gym
import numpy as np
from typing import Dict, List, Any, Optional
import random
from .consensus_env import ConsensusEnv
from ..core.node import NodeState

class ByzantineConsensusEnv(ConsensusEnv):
    """Environment specifically for Byzantine fault tolerance training."""
    
    def __init__(self, num_nodes=7, byzantine_ratio=0.3, max_steps=1000):
        super().__init__(num_nodes, byzantine_ratio, max_steps)
        self.byzantine_behavior_types = ['crash', 'equivocate', 'delay', 'corrupt']
        self.fault_history = []
        
    def reset(self):
        obs = super().reset()
        self.fault_history = []
        self._assign_byzantine_behaviors()
        return obs
    
    def _assign_byzantine_behaviors(self):
        """Assign specific Byzantine behaviors to faulty nodes."""
        for node_id, node in self.nodes.items():
            if node.is_byzantine:
                behavior = random.choice(self.byzantine_behavior_types)
                node.byzantine_behavior = behavior
                self.fault_history.append({
                    'node': node_id,
                    'behavior': behavior,
                    'step': self.current_step
                })
    
    def step(self, actions):
        # Apply Byzantine behaviors before normal step
        self._apply_byzantine_behaviors()
        
        obs, rewards, done, info = super().step(actions)
        
        # Additional Byzantine-specific rewards
        rewards = self._calculate_byzantine_rewards(rewards)
        
        # Add Byzantine-specific info
        info['fault_history'] = self.fault_history
        info['byzantine_behaviors'] = {
            nid: getattr(n, 'byzantine_behavior', None) 
            for nid, n in self.nodes.items() if n.is_byzantine
        }
        
        return obs, rewards, done, info
    
    def _apply_byzantine_behaviors(self):
        """Apply Byzantine behaviors to faulty nodes."""
        for node_id, node in self.nodes.items():
            if not node.is_byzantine:
                continue
                
            behavior = getattr(node, 'byzantine_behavior', 'crash')
            
            if behavior == 'crash':
                # Node stops responding
                node.state = NodeState.OFFLINE
                node.inbox.clear()
                node.outbox.clear()
                
            elif behavior == 'equivocate':
                # Send conflicting messages
                if node.state == NodeState.LEADER and len(node.log) > 0:
                    # Create conflicting log entries
                    fake_entry = f"fake_{random.randint(1000, 9999)}"
                    node.log.append(fake_entry)
                    
            elif behavior == 'delay':
                # Delay message processing
                if len(node.inbox) > 0 and random.random() < 0.7:
                    # Skip processing messages this round
                    node.inbox.clear()
                    
            elif behavior == 'corrupt':
                # Corrupt message contents
                for msg in node.outbox:
                    if random.random() < 0.5:
                        msg.payload = {'corrupted': True, 'original': msg.payload}
    
    def _calculate_byzantine_rewards(self, base_rewards):
        """Calculate rewards with Byzantine fault tolerance in mind."""
        rewards = base_rewards.copy()
        
        # Count honest nodes that reached consensus
        honest_consensus = 0
        byzantine_consensus = 0
        
        for node_id, node in self.nodes.items():
            if node.commit_index > 0:
                if node.is_byzantine:
                    byzantine_consensus += 1
                else:
                    honest_consensus += 1
        
        # Reward honest nodes for achieving consensus despite Byzantine nodes
        if honest_consensus > self.num_nodes // 2:
            for node_id, node in self.nodes.items():
                if not node.is_byzantine:
                    rewards[node_id] += 2.0
        
        # Penalize Byzantine nodes that disrupt consensus
        for node_id, node in self.nodes.items():
            if node.is_byzantine:
                if getattr(node, 'byzantine_behavior', None) in ['equivocate', 'corrupt']:
                    rewards[node_id] -= 1.0
        
        return rewards
    
    def make_byzantine(self, node_ids: List[str]):
        """Dynamically make nodes Byzantine."""
        for node_id in node_ids:
            if node_id in self.nodes:
                self.nodes[node_id].is_byzantine = True
                self._assign_byzantine_behaviors()
    
    def crash_nodes(self, node_ids: List[str]):
        """Crash specific nodes."""
        for node_id in node_ids:
            if node_id in self.nodes:
                self.nodes[node_id].state = NodeState.OFFLINE
                self.nodes[node_id].byzantine_behavior = 'crash'
                self.fault_history.append({
                    'node': node_id,
                    'behavior': 'crash',
                    'step': self.current_step
                })
    
    def slow_nodes(self, node_ids: List[str], factor: float = 0.5):
        """Make nodes slow to respond."""
        for node_id in node_ids:
            if node_id in self.nodes:
                self.nodes[node_id].byzantine_behavior = 'delay'
                self.nodes[node_id].delay_factor = factor