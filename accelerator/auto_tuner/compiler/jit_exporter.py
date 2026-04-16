import sys
import os
from typing import Dict, Any

# Ensure we can import the ledger
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from bkm_ledger import BKMLedger

class JitExporter:
    """
    Acts as the compiler frontend. When asked to compile a layer, it queries 
    the BKM JSON ledger. If the shape exists, it returns the optimized hardware 
    parameters. If not, it instantly returns a mathematically safe, unoptimized 
    fallback configuration.
    """
    def __init__(self, ledger_path: str = "bkm_ledger.json"):
        self.ledger = BKMLedger(filepath=ledger_path)

    def get_compilation_config(self, M: int, N: int, K: int, activation: int) -> Dict[str, Any]:
        """
        Returns the RL-optimized configuration if available, otherwise returns 
        the naive 4x4 fallback to ensure compilation NEVER hangs.
        """
        optimized_config = self.ledger.read_best_config(M, N, K, activation)
        
        if optimized_config is not None:
            # LOG: Cache Hit
            print(f"[JIT] BKM Cache Hit for {M}x{N}x{K} (IPC: {optimized_config['ipc']:.3f})")
            return optimized_config
        else:
            # LOG: Cache Miss, using safe fallback
            print(f"[JIT] BKM Cache Miss for {M}x{N}x{K}. Using safe 4x4 fallback.")
            return self._get_safe_fallback()

    def _get_safe_fallback(self) -> Dict[str, Any]:
        """
        Returns a mathematically safe configuration that is guaranteed to work 
        on the 4x4 Systolic Array, albeit slowly.
        """
        return {
            "tile_m": 4,          # Minimum valid systolic tile spatial height
            "tile_n": 4,          # Minimum valid systolic tile spatial width
            "burst_size": 16,     # Small, safe DMA burst size (256-bit bus = 16 bytes)
            "hardware_dataflow_mode": 0,  # Dense systolic (safe default)
            "ipc": 0.0            # Fallback marker
        }


if __name__ == "__main__":
    # Quick standalone test
    jit = JitExporter(ledger_path="/tmp/test_ledger.json")
    
    # 1. Test cache miss
    print("Testing Cache Miss:")
    config = jit.get_compilation_config(M=128, N=64, K=64, activation=1)
    print(config)
    
    # 2. Simulate the RL Daemon finding a better config
    print("\nSimulating RL Daemon Update:")
    updated = jit.ledger.update_if_better(M=128, N=64, K=64, activation=1, 
                                          tile_m=16, tile_n=32, burst_size=64, ipc=0.95)
    print(f"Ledger updated: {updated}")
    
    # 3. Test cache hit
    print("\nTesting Cache Hit:")
    config = jit.get_compilation_config(M=128, N=64, K=64, activation=1)
    print(config)
