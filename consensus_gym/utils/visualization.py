"""Visualization tools for consensus environments."""

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from typing import Dict, List, Any
import seaborn as sns

class ConsensusVisualizer:
    """Visualize consensus algorithm execution."""
    
    def __init__(self):
        self.fig = None
        self.axes = None
        sns.set_style("whitegrid")
        
    def plot_network_topology(self, nodes: Dict, partitions: List = None):
        """Visualize network topology and partitions."""
        G = nx.Graph()
        
        # Add nodes
        for node_id in nodes.keys():
            G.add_node(node_id)
        
        # Add edges (fully connected by default)
        node_list = list(nodes.keys())
        for i, node1 in enumerate(node_list):
            for node2 in node_list[i+1:]:
                # Check if nodes are partitioned
                if partitions:
                    partition1 = next((p for p in partitions if node1 in p), None)
                    partition2 = next((p for p in partitions if node2 in p), None)
                    if partition1 != partition2:
                        continue
                G.add_edge(node1, node2)
        
        # Set up colors
        node_colors = []
        for node_id, node in nodes.items():
            if node.state.value == 'LEADER':
                node_colors.append('red')
            elif node.is_byzantine:
                node_colors.append('black')
            elif node.state.value == 'OFFLINE':
                node_colors.append('gray')
            else:
                node_colors.append('lightblue')
        
        # Draw network
        plt.figure(figsize=(10, 8))
        pos = nx.spring_layout(G)
        nx.draw(G, pos, node_color=node_colors, with_labels=True,
                node_size=1000, font_size=10, font_weight='bold')
        
        plt.title("Network Topology")
        plt.axis('off')
        return plt.gcf()
    
    def plot_consensus_progress(self, history: List[Dict]):
        """Plot consensus progress over time."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        steps = [h['step'] for h in history]
        
        # Leader changes
        leader_changes = [h['leader_changes'] for h in history]
        axes[0, 0].plot(steps, leader_changes, 'b-')
        axes[0, 0].set_title('Leader Changes Over Time')
        axes[0, 0].set_xlabel('Step')
        axes[0, 0].set_ylabel('Total Leader Changes')
        
        # Commit index progress
        if 'nodes' in history[0]:
            for node_id in history[0]['nodes'].keys():
                commits = [h['nodes'][node_id].get('commit_index', 0) for h in history]
                axes[0, 1].plot(steps, commits, label=node_id, alpha=0.7)
            axes[0, 1].set_title('Commit Index Progress')
            axes[0, 1].set_xlabel('Step')
            axes[0, 1].set_ylabel('Commit Index')
            axes[0, 1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Messages in flight
        messages = [h.get('network', {}).get('messages_queued', 0) for h in history]
        axes[1, 0].plot(steps, messages, 'g-')
        axes[1, 0].set_title('Messages in Flight')
        axes[1, 0].set_xlabel('Step')
        axes[1, 0].set_ylabel('Message Count')
        
        # Node states distribution
        state_counts = {'FOLLOWER': [], 'CANDIDATE': [], 'LEADER': [], 'OFFLINE': []}
        for h in history:
            counts = {'FOLLOWER': 0, 'CANDIDATE': 0, 'LEADER': 0, 'OFFLINE': 0}
            for node_data in h.get('nodes', {}).values():
                state = node_data.get('state', 'FOLLOWER')
                counts[state] = counts.get(state, 0) + 1
            for state in state_counts:
                state_counts[state].append(counts.get(state, 0))
        
        bottom = np.zeros(len(steps))
        colors = {'FOLLOWER': 'blue', 'CANDIDATE': 'yellow', 
                  'LEADER': 'red', 'OFFLINE': 'gray'}
        
        for state, color in colors.items():
            axes[1, 1].bar(steps, state_counts[state], bottom=bottom,
                          label=state, color=color, alpha=0.7)
            bottom += np.array(state_counts[state])
        
        axes[1, 1].set_title('Node State Distribution')
        axes[1, 1].set_xlabel('Step')
        axes[1, 1].set_ylabel('Node Count')
        axes[1, 1].legend()
        
        plt.tight_layout()
        return fig
    
    def plot_message_flow(self, messages: List[Dict], time_window: float = 1.0):
        """Visualize message flow between nodes."""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Group messages by time window
        if not messages:
            return fig
            
        min_time = min(m['timestamp'] for m in messages)
        max_time = max(m['timestamp'] for m in messages)
        
        # Create time bins
        time_bins = np.arange(min_time, max_time + time_window, time_window)
        
        # Count messages per bin and type
        message_types = set(m.get('type', 'unknown') for m in messages)
        type_counts = {msg_type: np.zeros(len(time_bins)-1) for msg_type in message_types}
        
        for msg in messages:
            msg_type = msg.get('type', 'unknown')
            msg_time = msg['timestamp']
            bin_idx = np.searchsorted(time_bins, msg_time) - 1
            if 0 <= bin_idx < len(time_bins) - 1:
                type_counts[msg_type][bin_idx] += 1
        
        # Plot stacked bar chart
        bottom = np.zeros(len(time_bins)-1)
        colors = plt.cm.Set3(np.linspace(0, 1, len(message_types)))
        
        for (msg_type, counts), color in zip(type_counts.items(), colors):
            ax.bar(time_bins[:-1], counts, width=time_window*0.8,
                  bottom=bottom, label=msg_type, color=color)
            bottom += counts
        
        ax.set_xlabel('Time')
        ax.set_ylabel('Message Count')
        ax.set_title('Message Flow Over Time')
        ax.legend()
        
        return fig
    
    def plot_fault_timeline(self, fault_history: List[Dict]):
        """Visualize fault injection timeline."""
        if not fault_history:
            return None
            
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Group faults by node
        nodes = list(set(f['node'] for f in fault_history))
        node_positions = {node: i for i, node in enumerate(nodes)}
        
        # Color map for fault types
        fault_colors = {
            'byzantine': 'red',
            'crash': 'black',
            'slow': 'orange',
            'partition': 'purple',
            'equivocate': 'brown',
            'corrupt': 'darkred',
            'delay': 'yellow'
        }
        
        for fault in fault_history:
            node_pos = node_positions[fault['node']]
            step = fault['step']
            behavior = fault.get('behavior', 'unknown')
            color = fault_colors.get(behavior, 'gray')
            
            ax.scatter(step, node_pos, c=color, s=100, alpha=0.7, label=behavior)
        
        ax.set_yticks(range(len(nodes)))
        ax.set_yticklabels(nodes)
        ax.set_xlabel('Step')
        ax.set_ylabel('Node')
        ax.set_title('Fault Injection Timeline')
        
        # Remove duplicate labels
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc='upper right')
        
        return fig