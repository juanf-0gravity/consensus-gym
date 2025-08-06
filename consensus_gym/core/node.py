# consensus node stuff
# TODO: clean this up later, works for now

import numpy as np
from typing import Dict, List, Optional, Any
from enum import Enum
import time
import uuid
import random

# node states - pretty standard
class NodeState(Enum):
    FOLLOWER = "follower"
    CANDIDATE = "candidate" 
    LEADER = "leader"
    FAULTY = "faulty"  # byzantine
    OFFLINE = "offline"


class MessageType(Enum):
    HEARTBEAT = "heartbeat"
    VOTE_REQUEST = "vote_request" 
    VOTE_RESPONSE = "vote_response"
    APPEND_ENTRIES = "append_entries"
    APPEND_RESPONSE = "append_response"
    # TODO: add paxos messages later
    PREPARE = "prepare"
    PROMISE = "promise" 
    ACCEPT = "accept"
    ACCEPTED = "accepted"
    COMMIT = "commit"

class Message:
    def __init__(self, msg_type, sender, receiver, term=0, data=None):
        self.id = str(uuid.uuid4())  # probably overkill but whatever
        self.type = msg_type
        self.sender_id = sender
        self.receiver_id = receiver
        self.term = term
        self.data = data if data else {}
        self.timestamp = time.time()
        self.delivered = False

class LogEntry:
    def __init__(self, term, index, command, committed=False):
        self.term = term
        self.index = index
        self.command = command
        self.committed = committed
        self.timestamp = time.time()

class ConsensusNode:
    """Main consensus node - does raft-ish stuff but RL agents control it"""
    
    def __init__(self, node_id, cluster_size, byzantine_tolerance=1):
        self.node_id = node_id
        self.cluster_size = cluster_size
        self.byzantine_tolerance = byzantine_tolerance
        
        # basic raft state
        self.state = NodeState.FOLLOWER
        self.current_term = 0
        self.voted_for = None
        self.log = []
        self.commit_index = 0
        self.last_applied = 0
        
        # leader stuff (gets reset when not leader)
        self.next_index = {}
        self.match_index = {}
        
        # timing - these numbers are kinda arbitrary
        self.last_heartbeat = time.time()
        self.election_timeout = random.uniform(0.15, 0.3)  
        self.heartbeat_interval = 0.05
        
        # message handling
        self.inbox = []
        self.outbox = []
        
        # byzantine stuff for testing
        self.is_byzantine = False
        self.fault_probability = 0.0
        
        # stats - useful for debugging
        self.messages_sent = 0
        self.messages_received = 0
        self.elections_initiated = 0
        self.successful_commits = 0
        
        # keep track of other nodes
        self.peer_states = {}
        self.network_conditions = {
            'latency': 0.01,
            'packet_loss': 0.0,
            'partition_risk': 0.0
        }
    
    def get_observation(self):
        """Build observation vector for RL agent - this is kinda messy but works"""
        obs = []
        
        # encode current state as one-hot
        state_vec = [0] * 6
        if self.state == NodeState.FOLLOWER:
            state_vec[0] = 1
        elif self.state == NodeState.CANDIDATE:
            state_vec[1] = 1
        elif self.state == NodeState.LEADER:
            state_vec[2] = 1
        elif self.state == NodeState.FAULTY:
            state_vec[3] = 1
        elif self.state == NodeState.OFFLINE:
            state_vec[4] = 1
        state_vec[5] = 1 if self.is_byzantine else 0
        obs.extend(state_vec)
        
        # term and timing info
        now = time.time()
        obs.append(self.current_term / 100.0)  # normalize term
        obs.append((now - self.last_heartbeat) / self.election_timeout)
        obs.append(len(self.log) / 1000.0)  # normalize log size
        obs.append(self.commit_index / max(1, len(self.log)))
        
        # network stuff
        obs.append(self.network_conditions['latency'] * 100)
        obs.append(self.network_conditions['packet_loss'])
        obs.append(self.network_conditions['partition_risk'])
        
        # info about other nodes - this gets big with larger clusters
        for i in range(self.cluster_size):
            peer_id = f"node_{i}"
            if peer_id == self.node_id:
                obs.extend([1.0, self.current_term / 100.0, len(self.log) / 1000.0, 1.0])
            elif peer_id in self.peer_states:
                peer = self.peer_states[peer_id]
                obs.extend([
                    0.0,
                    peer.get('term', 0) / 100.0,
                    peer.get('log_length', 0) / 1000.0,
                    1.0 if peer.get('responsive', True) else 0.0
                ])
            else:
                obs.extend([0.0, 0.0, 0.0, 0.0])  # don't know about this peer
        
        # message queue sizes
        obs.append(len(self.inbox) / 10.0)
        obs.append(len(self.outbox) / 10.0)
        
        return np.array(obs, dtype=np.float32)
    
    def execute_action(self, action, peers):
        """Execute RL action - returns messages to send"""
        messages = []
        
        if action == 0:  # do nothing
            pass
            
        elif action == 1:  # send heartbeat (only if leader)
            if self.state == NodeState.LEADER:
                for peer in peers:
                    if peer != self.node_id:
                        msg = Message(
                            MessageType.HEARTBEAT,
                            self.node_id,
                            peer,
                            self.current_term,
                            {'commit_index': self.commit_index}
                        )
                        messages.append(msg)
                        
        elif action == 2:  # start election
            if self.state != NodeState.LEADER:
                self.start_election()
                for peer in peers:
                    if peer != self.node_id:
                        last_log_term = 0
                        if len(self.log) > 0:
                            last_log_term = self.log[-1].term
                        msg = Message(
                            MessageType.VOTE_REQUEST,
                            self.node_id,
                            peer,
                            self.current_term,
                            {
                                'last_log_index': len(self.log) - 1,
                                'last_log_term': last_log_term
                            }
                        )
                        messages.append(msg)
                        
        elif action == 3:  # replicate log
            if self.state == NodeState.LEADER and len(self.log) > 0:
                for peer in peers:
                    if peer != self.node_id:
                        next_idx = self.next_index.get(peer, 0)
                        prev_idx = next_idx - 1
                        prev_term = 0
                        
                        if prev_idx >= 0 and prev_idx < len(self.log):
                            prev_term = self.log[prev_idx].term
                            
                        entries_to_send = []
                        for entry in self.log[next_idx:]:
                            entries_to_send.append((entry.term, entry.index, entry.command))
                            
                        msg = Message(
                            MessageType.APPEND_ENTRIES,
                            self.node_id,
                            peer,
                            self.current_term,
                            {
                                'prev_log_index': prev_idx,
                                'prev_log_term': prev_term,
                                'entries': entries_to_send,
                                'leader_commit': self.commit_index
                            }
                        )
                        messages.append(msg)
                        
        elif action == 4:  # step down from leadership
            if self.state == NodeState.LEADER or self.state == NodeState.CANDIDATE:
                self.state = NodeState.FOLLOWER
                self.voted_for = None
                
        elif action == 5:  # commit entries
            if self.state == NodeState.LEADER:
                self.update_commit_index()
        
        # add more actions as needed...
        
        self.outbox.extend(messages)
        self.messages_sent += len(messages)
        return messages
    
    def process_message(self, msg):
        """Handle incoming message"""
        self.messages_received += 1
        self.inbox.append(msg)
        
        # byzantine nodes might do weird stuff
        if self.is_byzantine and random.random() < self.fault_probability:
            return self.byzantine_response(msg)
        
        response = None
        
        if msg.type == MessageType.VOTE_REQUEST:
            response = self.handle_vote_request(msg)
        elif msg.type == MessageType.VOTE_RESPONSE:
            self.handle_vote_response(msg)
        elif msg.type == MessageType.HEARTBEAT:
            self.handle_heartbeat(msg)
        elif msg.type == MessageType.APPEND_ENTRIES:
            response = self.handle_append_entries(msg)
        elif msg.type == MessageType.APPEND_RESPONSE:
            self.handle_append_response(msg)
        
        return response
    
    def start_election(self):
        """Become candidate and start election"""
        self.state = NodeState.CANDIDATE
        self.current_term += 1
        self.voted_for = self.node_id
        self.last_heartbeat = time.time()
        self.elections_initiated += 1
        self.election_timeout = random.uniform(0.15, 0.3)  # randomize timeout
    
    def handle_vote_request(self, msg):
        """Handle vote request - basic raft logic"""
        vote_granted = False
        
        if msg.term > self.current_term:
            self.current_term = msg.term
            self.voted_for = None
            self.state = NodeState.FOLLOWER
        
        if (msg.term == self.current_term and 
            (self.voted_for is None or self.voted_for == msg.sender_id)):
            
            # check if candidate's log is up to date
            candidate_last_idx = msg.data.get('last_log_index', -1)
            candidate_last_term = msg.data.get('last_log_term', 0)
            
            our_last_term = self.log[-1].term if len(self.log) > 0 else 0
            our_last_idx = len(self.log) - 1
            
            log_ok = (candidate_last_term > our_last_term or
                     (candidate_last_term == our_last_term and 
                      candidate_last_idx >= our_last_idx))
            
            if log_ok:
                vote_granted = True
                self.voted_for = msg.sender_id
        
        return Message(
            MessageType.VOTE_RESPONSE,
            self.node_id,
            msg.sender_id,
            self.current_term,
            {'vote_granted': vote_granted}
        )
    
    def handle_vote_response(self, msg):
        """Count votes and maybe become leader"""
        if (self.state == NodeState.CANDIDATE and 
            msg.term == self.current_term and
            msg.data.get('vote_granted', False)):
            
            # this is simplified vote counting
            # in real impl would track all votes properly
            if self.messages_received >= (self.cluster_size // 2):
                self.become_leader()
    
    def become_leader(self):
        """Transition to leader"""
        self.state = NodeState.LEADER
        # initialize leader state
        for i in range(self.cluster_size):
            peer_id = f"node_{i}"
            self.next_index[peer_id] = len(self.log)
            self.match_index[peer_id] = 0
    
    def handle_heartbeat(self, msg):
        """Reset election timer on heartbeat"""
        if msg.term >= self.current_term:
            self.current_term = msg.term
            self.state = NodeState.FOLLOWER
            self.voted_for = None
            self.last_heartbeat = time.time()
            
            # update commit index from leader
            leader_commit = msg.data.get('commit_index', 0)
            if leader_commit > self.commit_index:
                self.commit_index = min(leader_commit, len(self.log))
    
    def handle_append_entries(self, msg):
        """Handle log replication from leader"""
        success = False
        
        if msg.term >= self.current_term:
            self.current_term = msg.term
            self.state = NodeState.FOLLOWER
            self.last_heartbeat = time.time()
            
            prev_idx = msg.data.get('prev_log_index', -1)
            prev_term = msg.data.get('prev_log_term', 0)
            
            # check if prev entry matches
            if (prev_idx == -1 or 
                (prev_idx < len(self.log) and 
                 self.log[prev_idx].term == prev_term)):
                
                success = True
                entries = msg.data.get('entries', [])
                
                # append new entries
                for i, (term, idx, cmd) in enumerate(entries):
                    log_idx = prev_idx + 1 + i
                    entry = LogEntry(term, idx, cmd)
                    
                    if log_idx >= len(self.log):
                        self.log.append(entry)
                    else:
                        self.log[log_idx] = entry  # overwrite conflicting entry
                
                # update commit index
                leader_commit = msg.data.get('leader_commit', 0)
                if leader_commit > self.commit_index:
                    self.commit_index = min(leader_commit, len(self.log))
                    self.successful_commits += 1
        
        return Message(
            MessageType.APPEND_RESPONSE,
            self.node_id,
            msg.sender_id,
            self.current_term,
            {
                'success': success,
                'match_index': len(self.log) if success else 0
            }
        )
    
    def handle_append_response(self, msg):
        """Handle response to log replication"""
        if self.state == NodeState.LEADER and msg.term == self.current_term:
            if msg.data.get('success', False):
                # update follower's progress
                match_idx = msg.data.get('match_index', 0)
                self.match_index[msg.sender_id] = match_idx
                self.next_index[msg.sender_id] = match_idx + 1
            else:
                # decrement and retry
                current = self.next_index.get(msg.sender_id, 0)
                self.next_index[msg.sender_id] = max(0, current - 1)
    
    def update_commit_index(self):
        """Commit entries that are replicated to majority"""
        if self.state != NodeState.LEADER:
            return
            
        for n in range(self.commit_index + 1, len(self.log)):
            count = 1  # count ourselves
            for match_idx in self.match_index.values():
                if match_idx >= n:
                    count += 1
            
            # commit if majority has replicated
            if count > self.cluster_size // 2 and self.log[n].term == self.current_term:
                self.commit_index = n
                self.successful_commits += 1
    
    def byzantine_response(self, msg):
        """Generate malicious response for byzantine testing"""
        if random.random() < 0.5:
            return None  # drop message
        
        # send wrong info
        if msg.type == MessageType.VOTE_REQUEST:
            return Message(
                MessageType.VOTE_RESPONSE,
                self.node_id,
                msg.sender_id,
                self.current_term + random.randint(0, 2),  # wrong term
                {'vote_granted': random.choice([True, False])}
            )
        
        return None
    
    def add_log_entry(self, command):
        """Add new entry to log (leader only)"""
        if self.state == NodeState.LEADER:
            entry = LogEntry(self.current_term, len(self.log), command)
            self.log.append(entry)
            return True
        return False
    
    def is_election_timeout(self):
        """Check if we should start election"""
        return (time.time() - self.last_heartbeat) > self.election_timeout
    
    def get_stats(self):
        """Return node statistics"""
        return {
            'node_id': self.node_id,
            'state': self.state.value,
            'term': self.current_term,
            'log_len': len(self.log),
            'commit_idx': self.commit_index,
            'msgs_sent': self.messages_sent,
            'msgs_recv': self.messages_received,
            'elections': self.elections_initiated,
            'commits': self.successful_commits,
            'byzantine': self.is_byzantine
        }