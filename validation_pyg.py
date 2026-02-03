import sys
import torch
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data

def test_gnn_execution():
    print("--- PyG (ICLR 2019) Binary Integrity Check ---")
    try:
        # 1. Create a dummy graph
        edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
        x = torch.randn(2, 16)
        data = Data(x=x, edge_index=edge_index)

        # 2. Trigger the JIT/C++ layer
        # GCNConv calls torch-scatter/sparse internally.
        # This is where the ABI mismatch or ImportError will happen.
        print("--> Initializing GCNConv layer...")
        conv = GCNConv(16, 32)
        
        print("--> Executing forward pass...")
        out = conv(data.x, data.edge_index)
        
        if out.shape == (2, 32):
            print("    [✓] Forward pass successful.")
            print("--- SMOKE TEST PASSED ---")

    except (ImportError, RuntimeError) as e:
        print(f"CRITICAL BINARY ERROR: {str(e)}")
        # This is exactly what AURA needs to fix
        sys.exit(1)
    except Exception as e:
        print(f"VALIDATION FAILURE: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    test_gnn_execution()