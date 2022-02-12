import random
import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

@dataclass
class NetworkMessage:
    sender_id: int
    receiver_id: int
    message_type: str
    payload: Any
    timestamp: float
    delay: float = 0.0

class NetworkSimulator:
    def __init__(self, latency_range=(0.01, 0.1), packet_loss_rate=0.0, partition_probability=0.0):
        self.latency_range = latency_range
        self.packet_loss_rate = packet_loss_rate
        self.partition_probability = partition_probability
        self.message_queue: List[NetworkMessage] = []
        self.partitioned_nodes: set = set()
        self.current_time = 0.0
        
    def send_message(self, sender_id: int, receiver_id: int, message_type: str, payload: Any):
        if random.random() < self.packet_loss_rate:
            return
            
        if sender_id in self.partitioned_nodes or receiver_id in self.partitioned_nodes:
            return
            
        delay = random.uniform(*self.latency_range)
        message = NetworkMessage(
            sender_id=sender_id,
            receiver_id=receiver_id,
            message_type=message_type,
            payload=payload,
            timestamp=self.current_time,
            delay=delay
        )
        self.message_queue.append(message)
        
    def get_ready_messages(self, current_time: float) -> List[NetworkMessage]:
        self.current_time = current_time
        ready_messages = []
        remaining_messages = []
        
        for message in self.message_queue:
            if message.timestamp + message.delay <= current_time:
                ready_messages.append(message)
            else:
                remaining_messages.append(message)
                
        self.message_queue = remaining_messages
        return ready_messages
        
    def create_partition(self, node_ids: List[int]):
        self.partitioned_nodes.update(node_ids)
        
    def heal_partition(self):
        self.partitioned_nodes.clear()