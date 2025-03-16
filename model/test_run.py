# Import necessary libraries with threading limitations
import os
import sys
import numpy as np
import torch

# Set environment variables to limit threading
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

# Limit PyTorch threads
torch.set_num_threads(1)
if hasattr(torch, 'set_num_interop_threads'):
    torch.set_num_interop_threads(1)

# Import the rest of your modules
from data_processing import DrugEnzymeDataProcessor
from model import DrugEnzymeInteractionModel
import torch.nn as nn
import torch.optim as optim
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

# Create output directory
os.makedirs('output', exist_ok=True)

try:
    # Set device
    device = torch.device('cpu')  # Force CPU to avoid CUDA threading issues
    print(f"Using device: {device}")
    
    # Data processing with minimal operations
    processor = DrugEnzymeDataProcessor(
        panel_file='model_data/smiles_panel.csv',
        potency_file='model_data/smiles_potency.csv'
    )
    
    # Load and prepare data
    X_smiles, enzyme_encoding, y = processor.prepare_dataset()
    
    if len(X_smiles) == 0:
        print("No data available. Exiting.")
        sys.exit(1)
    
    # Print dataset statistics
    print(f"\nDataset statistics:")
    print(f"Number of samples: {len(X_smiles)}")
    print(f"Fingerprint dimension: {X_smiles.shape[1]}")
    print(f"Enzyme encoding shape: {enzyme_encoding.shape}")
    print(f"Potency range: {np.min(y):.2f} to {np.max(y):.2f}")
    
    # Initialize a simple model for testing
    model = DrugEnzymeInteractionModel(
        drug_input_dim=X_smiles.shape[1],
        enzyme_input_dim=enzyme_encoding.shape[1],
        enzyme_seq_length=enzyme_encoding.shape[0],
        hidden_dim=128,  # Reduced from 512
        dropout_rate=0.2
    )
    
    # Print model summary
    print("\nModel initialized successfully")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Test a forward pass with a small batch
    print("\nTesting forward pass...")
    test_drug = torch.FloatTensor(X_smiles[:2])
    test_enzyme = torch.FloatTensor(enzyme_encoding).unsqueeze(0).repeat(2, 1, 1)
    with torch.no_grad():
        output = model(test_drug, test_enzyme)
    print(f"Forward pass output shape: {output.shape}")
    
    print("\nTest completed successfully!")
    
except Exception as e:
    print(f"Error in test run: {e}")
    import traceback
    traceback.print_exc() 