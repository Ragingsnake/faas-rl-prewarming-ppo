"""
metrics_logger.py — Log agent metrics to JSON for later visualization.
"""
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List

log = logging.getLogger(__name__)


class MetricsLogger:
    """Collect step-by-step metrics for visualization."""
    
    def __init__(self, output_path: str = "checkpoints/metrics.json"):
        self.output_path = Path(output_path)
        try:
            self.output_path.parent.mkdir(parents=True, exist_ok=True)
        except PermissionError:
            log.warning("Cannot create %s, metrics will not be persisted", self.output_path.parent)
        self.metrics: List[Dict[str, Any]] = []
    
    def log_step(self, step: int, **kwargs):
        """Log a single step with arbitrary kwargs."""
        record = {"step": step, "ts": time.time(), **kwargs}
        self.metrics.append(record)
    
    def save(self):
        """Persist metrics to JSON."""
        with open(self.output_path, "w") as f:
            json.dump(self.metrics, f, indent=2)
        log.info("Metrics saved to %s (%d steps)", self.output_path, len(self.metrics))
    
    def __len__(self) -> int:
        return len(self.metrics)

    def reset(self):
        """Clear in-memory metrics."""
        self.metrics = []
