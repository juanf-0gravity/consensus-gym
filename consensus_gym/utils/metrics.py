"""Metrics collection and analysis for consensus environments."""

import numpy as np
from typing import Dict, List, Any, Optional
from collections import defaultdict, deque
import time

class MetricsCollector:
    """Collect and analyze metrics from consensus environments."""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.metrics = defaultdict(lambda: deque(maxlen=window_size))
        self.episode_metrics = []
        self.start_time = time.time()
        
    def record(self, metric_name: str, value: float, step: Optional[int] = None):
        """Record a single metric value."""
        self.metrics[metric_name].append({
            'value': value,
            'step': step,
            'timestamp': time.time() - self.start_time
        })
    
    def record_episode(self, episode_data: Dict[str, Any]):
        """Record metrics for an entire episode."""
        self.episode_metrics.append({
            **episode_data,
            'timestamp': time.time() - self.start_time
        })
    
    def get_metric(self, metric_name: str) -> List[float]:
        """Get all values for a specific metric."""
        return [m['value'] for m in self.metrics[metric_name]]
    
    def get_statistics(self, metric_name: str) -> Dict[str, float]:
        """Get statistics for a specific metric."""
        values = self.get_metric(metric_name)
        if not values:
            return {}
        
        return {
            'mean': np.mean(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values),
            'median': np.median(values),
            'count': len(values)
        }
    
    def get_all_statistics(self) -> Dict[str, Dict[str, float]]:
        """Get statistics for all metrics."""
        return {
            metric: self.get_statistics(metric)
            for metric in self.metrics.keys()
        }
    
    def analyze_consensus_performance(self, env_history: List[Dict]) -> Dict[str, Any]:
        """Analyze consensus performance from environment history."""
        if not env_history:
            return {}
        
        analysis = {
            'consensus_time': None,
            'leader_stability': 0,
            'message_efficiency': 0,
            'fault_tolerance': 0,
            'throughput': 0
        }
        
        # Find consensus time
        for i, state in enumerate(env_history):
            if self._check_consensus(state):
                analysis['consensus_time'] = i
                break
        
        # Leader stability (fewer changes is better)
        leader_changes = [s.get('leader_changes', 0) for s in env_history]
        if leader_changes:
            analysis['leader_stability'] = 1.0 / (1.0 + leader_changes[-1])
        
        # Message efficiency
        total_messages = sum(s.get('network', {}).get('messages_queued', 0) 
                           for s in env_history)
        if total_messages > 0:
            analysis['message_efficiency'] = len(env_history) / total_messages
        
        # Fault tolerance (ratio of successful commits despite faults)
        byzantine_nodes = env_history[0].get('byzantine_nodes', [])
        if byzantine_nodes:
            commits = sum(1 for s in env_history if self._check_consensus(s))
            analysis['fault_tolerance'] = commits / len(env_history)
        
        # Throughput (commits per step)
        total_commits = sum(s.get('total_commits', 0) for s in env_history)
        analysis['throughput'] = total_commits / len(env_history) if env_history else 0
        
        return analysis
    
    def _check_consensus(self, state: Dict) -> bool:
        """Check if consensus was reached in a given state."""
        if 'nodes' not in state:
            return False
            
        commit_indices = []
        for node_data in state['nodes'].values():
            if node_data.get('state') != 'OFFLINE':
                commit_indices.append(node_data.get('commit_index', 0))
        
        # Consensus reached if majority have same commit index > 0
        if commit_indices:
            most_common = max(set(commit_indices), key=commit_indices.count)
            count = commit_indices.count(most_common)
            return most_common > 0 and count > len(commit_indices) / 2
        
        return False
    
    def calculate_byzantine_metrics(self, env_history: List[Dict]) -> Dict[str, float]:
        """Calculate Byzantine-specific metrics."""
        metrics = {
            'byzantine_detection_rate': 0,
            'false_positive_rate': 0,
            'consensus_despite_byzantine': 0,
            'byzantine_impact': 0
        }
        
        if not env_history:
            return metrics
        
        # Track Byzantine behavior impact
        byzantine_nodes = set()
        detected_byzantine = set()
        false_positives = set()
        
        for state in env_history:
            # Get actual Byzantine nodes
            actual_byzantine = set(state.get('byzantine_nodes', []))
            byzantine_nodes.update(actual_byzantine)
            
            # Simulate detection (nodes with unusual behavior)
            for node_id, node_data in state.get('nodes', {}).items():
                if node_data.get('messages_sent', 0) > 50:  # Unusual activity
                    detected_byzantine.add(node_id)
        
        # Calculate detection metrics
        if byzantine_nodes:
            true_positives = detected_byzantine & byzantine_nodes
            metrics['byzantine_detection_rate'] = len(true_positives) / len(byzantine_nodes)
            
            false_positives = detected_byzantine - byzantine_nodes
            total_honest = len(env_history[0].get('nodes', {})) - len(byzantine_nodes)
            if total_honest > 0:
                metrics['false_positive_rate'] = len(false_positives) / total_honest
        
        # Check consensus achievement despite Byzantine nodes
        consensus_count = sum(1 for s in env_history if self._check_consensus(s))
        metrics['consensus_despite_byzantine'] = consensus_count / len(env_history)
        
        # Measure Byzantine impact on performance
        if len(env_history) > 1:
            with_byzantine = [s for s in env_history if s.get('byzantine_nodes')]
            without_byzantine = [s for s in env_history if not s.get('byzantine_nodes')]
            
            if with_byzantine and without_byzantine:
                avg_commits_with = np.mean([s.get('total_commits', 0) for s in with_byzantine])
                avg_commits_without = np.mean([s.get('total_commits', 0) for s in without_byzantine])
                
                if avg_commits_without > 0:
                    metrics['byzantine_impact'] = 1.0 - (avg_commits_with / avg_commits_without)
        
        return metrics
    
    def export_metrics(self, filepath: str):
        """Export metrics to a file."""
        import json
        
        export_data = {
            'metrics': {k: list(v) for k, v in self.metrics.items()},
            'episode_metrics': self.episode_metrics,
            'statistics': self.get_all_statistics(),
            'runtime': time.time() - self.start_time
        }
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)