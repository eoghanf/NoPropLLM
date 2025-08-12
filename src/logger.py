"""
Training logger for detailed metrics tracking.
"""

import json
import csv
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional
import torch


class TrainingLogger:
    """
    Logger for tracking training metrics with wall-clock time.
    Saves both JSON and CSV formats for easy analysis.
    """
    
    def __init__(
        self, 
        experiment_name: str,
        log_dir: str = "training_logs",
        config: Optional[Dict] = None
    ):
        """
        Initialize the training logger.
        
        Args:
            experiment_name: Name of the experiment (used for file naming)
            log_dir: Directory to save logs
            config: Optional config dictionary to save with logs
        """
        self.experiment_name = experiment_name
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create unique log files with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_prefix = f"{experiment_name}_{timestamp}"
        
        # File paths
        self.csv_path = self.log_dir / f"{self.log_prefix}_metrics.csv"
        self.json_path = self.log_dir / f"{self.log_prefix}_log.json"
        self.config_path = self.log_dir / f"{self.log_prefix}_config.json"
        
        # Timing
        self.start_time = time.time()
        
        # Data storage
        self.metrics = []
        self.batch_count = 0
        self.epoch_count = 0
        
        # Save config if provided
        if config is not None:
            self._save_config(config)
        
        # Initialize CSV file with headers
        self._initialize_csv()
        
        print(f"Training logger initialized:")
        print(f"  Metrics CSV: {self.csv_path}")
        print(f"  Full log JSON: {self.json_path}")
        print(f"  Config: {self.config_path}")
    
    def _save_config(self, config: Dict):
        """Save configuration to JSON file."""
        config_dict = {}
        for key, value in config.items():
            # Handle non-serializable types
            if hasattr(value, '__dict__'):
                config_dict[key] = str(value)
            elif torch.is_tensor(value):
                config_dict[key] = f"Tensor({value.shape})"
            else:
                try:
                    json.dumps(value)  # Test if serializable
                    config_dict[key] = value
                except (TypeError, ValueError):
                    config_dict[key] = str(value)
        
        with open(self.config_path, 'w') as f:
            json.dump(config_dict, f, indent=2)
    
    def _initialize_csv(self):
        """Initialize CSV file with headers."""
        headers = [
            'batch', 'epoch', 'wall_time_seconds', 'wall_time_minutes',
            'train_loss', 'val_loss', 'val_accuracy', 'learning_rate',
            'batch_time_seconds', 'samples_per_second'
        ]
        
        with open(self.csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(headers)
    
    def log_batch_metrics(
        self,
        epoch: int,
        batch_idx: int,
        train_loss: float,
        val_loss: float,
        val_accuracy: float,
        learning_rate: Optional[float] = None,
        batch_size: int = 1,
        batch_time: Optional[float] = None
    ):
        """
        Log metrics for a single batch.
        
        Args:
            epoch: Current epoch number
            batch_idx: Current batch index within epoch
            train_loss: Training loss for this batch
            val_loss: Validation loss (computed on validation set)
            val_accuracy: Validation accuracy
            learning_rate: Current learning rate
            batch_size: Batch size for throughput calculation
            batch_time: Time taken for this batch (seconds)
        """
        current_time = time.time()
        wall_time_seconds = current_time - self.start_time
        wall_time_minutes = wall_time_seconds / 60.0
        
        # Calculate throughput
        samples_per_second = batch_size / batch_time if batch_time and batch_time > 0 else 0
        
        # Create metric entry
        metric_entry = {
            'batch': self.batch_count,
            'global_batch': batch_idx,
            'epoch': epoch,
            'wall_time_seconds': wall_time_seconds,
            'wall_time_minutes': wall_time_minutes,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_accuracy': val_accuracy,
            'learning_rate': learning_rate,
            'batch_time_seconds': batch_time,
            'samples_per_second': samples_per_second,
            'timestamp': datetime.now().isoformat()
        }
        
        # Store in memory
        self.metrics.append(metric_entry)
        
        # Write to CSV immediately for real-time monitoring
        self._append_to_csv(metric_entry)
        
        # Save JSON periodically (every 100 batches to avoid too much I/O)
        if self.batch_count % 100 == 0:
            self._save_json()
        
        self.batch_count += 1
    
    def _append_to_csv(self, metric_entry: Dict):
        """Append a metric entry to the CSV file."""
        with open(self.csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            row = [
                metric_entry['batch'],
                metric_entry['epoch'],
                f"{metric_entry['wall_time_seconds']:.2f}",
                f"{metric_entry['wall_time_minutes']:.2f}",
                f"{metric_entry['train_loss']:.6f}",
                f"{metric_entry['val_loss']:.6f}",
                f"{metric_entry['val_accuracy']:.2f}",
                f"{metric_entry['learning_rate']:.8f}" if metric_entry['learning_rate'] else "",
                f"{metric_entry['batch_time_seconds']:.3f}" if metric_entry['batch_time_seconds'] else "",
                f"{metric_entry['samples_per_second']:.1f}"
            ]
            writer.writerow(row)
    
    def _save_json(self):
        """Save all metrics to JSON file."""
        log_data = {
            'experiment_name': self.experiment_name,
            'start_time': datetime.fromtimestamp(self.start_time).isoformat(),
            'total_batches': len(self.metrics),
            'total_epochs': self.epoch_count,
            'metrics': self.metrics
        }
        
        with open(self.json_path, 'w') as f:
            json.dump(log_data, f, indent=2)
    
    def log_epoch_end(self, epoch: int):
        """Mark the end of an epoch."""
        self.epoch_count = epoch
        # Force save JSON at epoch end
        self._save_json()
    
    def finalize(self):
        """Finalize logging and save all data."""
        print(f"\nFinalizing training logs...")
        
        # Final save
        self._save_json()
        
        # Print summary
        if self.metrics:
            total_time = self.metrics[-1]['wall_time_minutes']
            print(f"  Total training time: {total_time:.2f} minutes")
            print(f"  Total batches logged: {len(self.metrics)}")
            print(f"  Final validation accuracy: {self.metrics[-1]['val_accuracy']:.2f}%")
        
        print(f"  Logs saved to: {self.log_dir}")
    
    def get_current_metrics(self) -> Dict:
        """Get the most recent metrics."""
        if self.metrics:
            return self.metrics[-1].copy()
        return {}
    
    def get_experiment_summary(self) -> Dict:
        """Get a summary of the experiment."""
        if not self.metrics:
            return {}
        
        return {
            'experiment_name': self.experiment_name,
            'total_batches': len(self.metrics),
            'total_epochs': self.epoch_count,
            'total_time_minutes': self.metrics[-1]['wall_time_minutes'] if self.metrics else 0,
            'best_val_accuracy': max(m['val_accuracy'] for m in self.metrics),
            'final_val_accuracy': self.metrics[-1]['val_accuracy'],
            'final_train_loss': self.metrics[-1]['train_loss'],
            'final_val_loss': self.metrics[-1]['val_loss'],
            'log_files': {
                'csv': str(self.csv_path),
                'json': str(self.json_path),
                'config': str(self.config_path)
            }
        }


class BatchTimer:
    """Simple timer for measuring batch processing time."""
    
    def __init__(self):
        self.start_time = None
        
    def start(self):
        """Start timing."""
        self.start_time = time.time()
        
    def stop(self) -> float:
        """Stop timing and return elapsed time in seconds."""
        if self.start_time is None:
            return 0.0
        elapsed = time.time() - self.start_time
        self.start_time = None
        return elapsed