# main consensus environment 
# trying to make this work with pettingzoo for multi agent stuff

import gym
from gym import spaces
import numpy as np
import random
from typing import Dict, List, Any
import time

from ..core.node import ConsensusNode, MessageType, NodeState

class ConsensusEnv(gym.Env):
    """Basic consensus environment - agents learn to coordinate"""
    
    def __init__(self, num_nodes=5, byzantine_ratio=0.2, max_steps=1000):
        super().__init__()
        
        self.num_nodes = num_nodes
        self.byzantine_count = int(num_nodes * byzantine_ratio)
        self.max_steps = max_steps
        self.current_step = 0
        
        # create nodes
        self.nodes = {}
        for i in range(num_nodes):
            node_id = f"node_{i}"
            self.nodes[node_id] = ConsensusNode(node_id, num_nodes)
            
        # randomly make some byzantine
        byzantine_nodes = random.sample(list(self.nodes.keys()), self.byzantine_count)
        for node_id in byzantine_nodes:
            self.nodes[node_id].is_byzantine = True
            self.nodes[node_id].fault_probability = 0.3
        
        # observation space - each node gets its own obs
        # this is rough estimate, actual size depends on cluster size
        obs_size = 15 + (num_nodes * 4) + 2  # from node.get_observation()
        self.observation_space = spaces.Box(
            low=-1.0, high=10.0, shape=(obs_size,), dtype=np.float32
        )
        
        # action space - discrete actions for each node
        self.action_space = spaces.Discrete(6)  # simplified action set
        
        # network conditions
        self.network_latency = 0.01
        self.packet_loss_rate = 0.0
        self.partition_active = False
        
        # tracking stuff
        self.message_queue = []  # messages in flight
        self.committed_entries = []
        self.leader_changes = 0
        self.last_leader = None
        
    def reset(self):
        """Reset environment"""
        self.current_step = 0
        
        # reset all nodes
        for node in self.nodes.values():
            node.state = NodeState.FOLLOWER
            node.current_term = 0
            node.voted_for = None
            node.log = []
            node.commit_index = 0
            node.last_applied = 0
            node.inbox = []
            node.outbox = []
            node.last_heartbeat = time.time()
            node.messages_sent = 0
            node.messages_received = 0
            node.elections_initiated = 0
            node.successful_commits = 0
        
        # clear tracking
        self.message_queue = []
        self.committed_entries = []
        self.leader_changes = 0
        self.last_leader = None
        
        # add some initial log entries to make things interesting
        leader_candidate = random.choice(list(self.nodes.keys()))
        self.nodes[leader_candidate].state = NodeState.LEADER
        self.nodes[leader_candidate].add_log_entry("init_command_1")
        self.nodes[leader_candidate].add_log_entry("init_command_2")
        
        return self.get_observations()
    
    def step(self, actions):
        """Step the environment - actions is dict mapping node_id to action"""
        self.current_step += 1
        
        # apply actions for each node
        for node_id, action in actions.items():
            if node_id in self.nodes:
                peer_ids = [nid for nid in self.nodes.keys() if nid != node_id]
                messages = self.nodes[node_id].execute_action(action, peer_ids)
                
                # add messages to network queue with delay simulation
                for msg in messages:
                    # simulate network conditions
                    if random.random() > self.packet_loss_rate:
                        delay = self.network_latency + random.uniform(0, 0.01)
                        delivery_time = time.time() + delay
                        self.message_queue.append((delivery_time, msg))
        
        # deliver messages that are ready
        current_time = time.time()
        delivered_messages = []
        remaining_messages = []
        
        for delivery_time, msg in self.message_queue:
            if current_time >= delivery_time:
                delivered_messages.append(msg)
            else:
                remaining_messages.append((delivery_time, msg))
        
        self.message_queue = remaining_messages
        
        # process delivered messages
        for msg in delivered_messages:
            if msg.receiver_id in self.nodes:
                response = self.nodes[msg.receiver_id].process_message(msg)
                if response:
                    # add response to queue
                    delay = self.network_latency + random.uniform(0, 0.01) 
                    delivery_time = time.time() + delay
                    self.message_queue.append((delivery_time, response))
        
        # check for leader changes
        current_leader = None
        for node_id, node in self.nodes.items():
            if node.state == NodeState.LEADER:
                current_leader = node_id
                break
        
        if current_leader != self.last_leader:
            self.leader_changes += 1
            self.last_leader = current_leader
        
        # randomly add new log entries if we have a leader
        if current_leader and random.random() < 0.1:
            cmd = f"command_{self.current_step}_{random.randint(1, 100)}"
            self.nodes[current_leader].add_log_entry(cmd)
        
        # collect observations
        observations = self.get_observations()
        
        # calculate rewards
        rewards = self.calculate_rewards()
        
        # check if done
        done = self.current_step >= self.max_steps
        
        # info dict
        info = {
            'leader': current_leader,
            'leader_changes': self.leader_changes,
            'total_commits': sum(n.successful_commits for n in self.nodes.values()),
            'messages_in_flight': len(self.message_queue),
            'byzantine_nodes': [nid for nid, n in self.nodes.items() if n.is_byzantine]
        }
        
        return observations, rewards, done, info
    
    def get_observations(self):
        """Get observations for all nodes"""
        obs = {}
        for node_id, node in self.nodes.items():
            # update network conditions in node
            node.network_conditions = {
                'latency': self.network_latency,
                'packet_loss': self.packet_loss_rate,
                'partition_risk': 1.0 if self.partition_active else 0.0
            }
            obs[node_id] = node.get_observation()
        return obs
    
    def calculate_rewards(self):
        """Calculate rewards for each node - this is the tricky part"""
        rewards = {}
        
        # base reward - small negative to encourage efficiency
        base_reward = -0.01
        
        for node_id, node in self.nodes.items():
            reward = base_reward
            
            # reward for successful commits
            reward += node.successful_commits * 1.0
            
            # small penalty for excessive messaging
            if node.messages_sent > 10:
                reward -= 0.1 * (node.messages_sent - 10)
            
            # penalty for byzantine behavior (for non-byzantine nodes)
            if not node.is_byzantine:
                # reward for being responsive
                if node.state != NodeState.OFFLINE:
                    reward += 0.1
                
                # reward for participating in consensus
                if node.messages_received > 0:
                    reward += 0.05
            
            # penalty for network partitions if we can't reach consensus
            if self.partition_active:
                reward -= 0.2
            
            rewards[node_id] = reward
        
        return rewards
    
    def render(self, mode='human'):
        """Simple text rendering of current state"""
        if mode == 'human':
            print(f"\n=== Consensus Step {self.current_step} ===")
            
            # show node states
            states = {}
            for node_id, node in self.nodes.items():
                state_info = f"{node.state.value}"
                if node.is_byzantine:
                    state_info += " (BYZ)"
                if node.state == NodeState.LEADER:
                    state_info += f" term:{node.current_term}"
                states[node_id] = state_info
            
            print("Node States:", states)
            
            # show leader if any
            leader = None
            for node_id, node in self.nodes.items():
                if node.state == NodeState.LEADER:
                    leader = node_id
                    break
            
            if leader:
                leader_node = self.nodes[leader]
                print(f"Leader: {leader} (term {leader_node.current_term}, log size {len(leader_node.log)})")
            else:
                print("No current leader")
            
            print(f"Messages in flight: {len(self.message_queue)}")
            print(f"Leader changes so far: {self.leader_changes}")
            
    def inject_network_partition(self, duration=5.0):
        """Simulate network partition for testing"""
        # TODO: implement actual partitioning logic
        self.partition_active = True
        self.packet_loss_rate = 0.5  # simulate partition with high packet loss
        
        # could use threading.Timer to auto-restore, but keeping it simple for now
        print(f"Network partition injected! Duration: {duration}s")
    
    def restore_network(self):
        """Restore normal network conditions"""
        self.partition_active = False
        self.packet_loss_rate = 0.0
        print("Network restored to normal conditions")
    
    def get_stats(self):
        """Get detailed stats for analysis"""
        stats = {
            'step': self.current_step,
            'leader_changes': self.leader_changes,
            'current_leader': None,
            'nodes': {}
        }
        
        for node_id, node in self.nodes.items():
            stats['nodes'][node_id] = node.get_stats()
            if node.state == NodeState.LEADER:
                stats['current_leader'] = node_id
        
        stats['network'] = {
            'latency': self.network_latency,
            'packet_loss': self.packet_loss_rate,
            'partition_active': self.partition_active,
            'messages_queued': len(self.message_queue)
        }
        
        return stats