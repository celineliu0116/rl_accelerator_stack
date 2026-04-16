import json
import os
import fcntl
from typing import Dict, Any, Optional

class BKMLedger:
    """
    Thread-safe and process-safe Best Known Method (BKM) Ledger.
    Reads and writes hardware configurations atomically to prevent corruption 
    between the background RL Daemon and the JIT Exporter loop.
    """
    def __init__(self, filepath: str = "bkm_ledger.json"):
        self.filepath = filepath
        self._ensure_exists()

    def _ensure_exists(self):
        """Creates an empty ledger if one does not exist."""
        if not os.path.exists(self.filepath):
            with open(self.filepath, 'w') as f:
                json.dump({}, f)

    def _get_layer_key(self, M: int, N: int, K: int, activation: int,
                       workload_tag: str = "", sparsity_bucket: int = -1) -> str:
        """Standardized string key for the JSON dictionary."""
        base = f"{M}x{N}x{K}_act{activation}"
        if workload_tag:
            if sparsity_bucket >= 0:
                return f"{base}_{workload_tag}_sp{sparsity_bucket}"
            return f"{base}_{workload_tag}"
        return base

    def read_best_config(self, M: int, N: int, K: int, activation: int,
                         workload_tag: str = "", sparsity_bucket: int = -1) -> Optional[Dict[str, Any]]:
        """
        Atomically reads the ledger and returns the best known configuration 
        for the exact layer geometry, if it exists.
        """
        layer_key = self._get_layer_key(M, N, K, activation, workload_tag, sparsity_bucket)
        
        try:
            with open(self.filepath, 'r') as f:
                # Shared lock for reading
                fcntl.flock(f, fcntl.LOCK_SH)
                try:
                    ledger = json.load(f)
                finally:
                    fcntl.flock(f, fcntl.LOCK_UN)
            
            return ledger.get(layer_key)
            
        except (json.JSONDecodeError, FileNotFoundError):
            return None

    def update_if_better(self, M: int, N: int, K: int, activation: int,
                         tile_m: int, tile_n: int, burst_size: int,
                         prefetch_depth: int, tile_b: int,
                         hardware_dataflow_mode: int = 0, ipc: float = 0.0,
                         workload_tag: str = "", sparsity_bucket: int = -1) -> bool:
        """
        Atomically updates the ledger with a new configuration ONLY IF the 
        provided IPC is strictly greater than the existing record.
        Returns True if updated, False otherwise.
        """
        layer_key = self._get_layer_key(M, N, K, activation, workload_tag, sparsity_bucket)
        new_entry = {
            "tile_m": int(tile_m),
            "tile_n": int(tile_n),
            "burst_size": int(burst_size),
            "prefetch_depth": int(prefetch_depth),
            "tile_b": int(tile_b),
            "hardware_dataflow_mode": int(hardware_dataflow_mode),
            "ipc": float(ipc),
            "workload_tag": str(workload_tag),
            "sparsity_bucket": int(sparsity_bucket),
        }

        with open(self.filepath, 'r+') as f:
            # Exclusive lock for writing
            fcntl.flock(f, fcntl.LOCK_EX)
            try:
                # Read current state while locked
                try:
                    ledger = json.load(f)
                except json.JSONDecodeError:
                    ledger = {}

                current_entry = ledger.get(layer_key)
                
                # Check if we should update
                if current_entry is None or ipc > current_entry.get("ipc", -1.0):
                    ledger[layer_key] = new_entry
                    
                    # Atomic write-back
                    f.seek(0)
                    f.truncate()
                    json.dump(ledger, f, indent=4)
                    return True
                
                return False
                
            finally:
                fcntl.flock(f, fcntl.LOCK_UN)
