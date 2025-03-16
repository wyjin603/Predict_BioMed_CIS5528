import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import os
import sys

from data_processing import DrugEnzymeDataProcessor, DrugEnzymeDataset
from model import DrugEnzymeInteractionModel

# Set environment variables to limit threading
os.environ["OMP_NUM_THREADS"] = "1"  # OpenMP
os.environ["MKL_NUM_THREADS"] = "1"  # MKL
os.environ["NUMEXPR_NUM_THREADS"] = "1"  # NumExpr
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"  # Accelerate
os.environ["OPENBLAS_NUM_THREADS"] = "1"  # OpenBLAS

# Limit PyTorch threads
torch.set_num_threads(1)
if hasattr(torch, 'set_num_interop_threads'):
    torch.set_num_interop_threads(1)

# Create output directory
os.makedirs('output', exist_ok=True)

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=100, device='cuda'):
    """
    Train the drug-enzyme interaction model.
    
    Args:
        model: The neural network model
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        criterion: Loss function
        optimizer: Optimizer
        num_epochs: Number of training epochs
        device: Device to train on ('cuda' or 'cpu')
        
    Returns:
        Trained model and training history
    """
    model = model.to(device)
    history = {'train_loss': [], 'val_loss': []}
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        
        for batch in train_loader:
            drug = batch['drug'].to(device)
            enzyme = batch['enzyme'].to(device)
            potency = batch['potency'].to(device)
            
            # Zero the gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(drug, enzyme)
            loss = criterion(outputs, potency)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * drug.size(0)
        
        train_loss = train_loss / len(train_loader.dataset)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for batch in val_loader:
                drug = batch['drug'].to(device)
                enzyme = batch['enzyme'].to(device)
                potency = batch['potency'].to(device)
                
                outputs = model(drug, enzyme)
                loss = criterion(outputs, potency)
                
                val_loss += loss.item() * drug.size(0)
                
        val_loss = val_loss / len(val_loader.dataset)
        
        # Record history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        
        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        
        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
            }, f'output/checkpoint_epoch_{epoch+1}.pt')
    
    return model, history

def evaluate_model(model, test_loader, device='cuda'):
    """
    Evaluate the model on test data.
    
    Args:
        model: Trained model
        test_loader: DataLoader for test data
        device: Device to evaluate on
        
    Returns:
        Dictionary of evaluation metrics
    """
    model.eval()
    y_true = []
    y_pred = []
    
    with torch.no_grad():
        for batch in test_loader:
            drug = batch['drug'].to(device)
            enzyme = batch['enzyme'].to(device)
            potency = batch['potency'].to(device)
            
            outputs = model(drug, enzyme)
            
            y_true.extend(potency.cpu().numpy())
            y_pred.extend(outputs.cpu().numpy())
    
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    
    return {
        'mse': mse,
        'rmse': rmse,
        'r2': r2,
        'y_true': y_true,
        'y_pred': y_pred
    }

def plot_results(history, evaluation_results):
    """Plot training history and prediction results."""
    # Plot training history
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training History')
    
    # Plot predictions vs actual
    plt.subplot(1, 2, 2)
    plt.scatter(evaluation_results['y_true'], evaluation_results['y_pred'], alpha=0.5)
    plt.plot([min(evaluation_results['y_true']), max(evaluation_results['y_true'])], 
             [min(evaluation_results['y_true']), max(evaluation_results['y_true'])], 
             'r--')
    plt.xlabel('Actual Potency')
    plt.ylabel('Predicted Potency')
    plt.title(f'Predictions (R² = {evaluation_results["r2"]:.3f}, RMSE = {evaluation_results["rmse"]:.3f})')
    
    plt.tight_layout()
    plt.savefig('output/model_results.png')
    print("Results plot saved to 'output/model_results.png'")


try:
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Data processing
    processor = DrugEnzymeDataProcessor(
        panel_file='model_data/smiles_panel.csv',
        potency_file='model_data/smiles_potency.csv'
    )
    
    X_smiles, enzyme_encoding, y = processor.prepare_dataset()
    
    # Print dataset statistics
    print(f"\nDataset statistics:")
    print(f"Number of samples: {len(X_smiles)}")
    print(f"Fingerprint dimension: {X_smiles.shape[1]}")
    print(f"Enzyme encoding shape: {enzyme_encoding.shape}")
    print(f"Potency range: {y.min():.2f} to {y.max():.2f}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X_smiles, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)
    
    print(f"\nData split:")
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Test samples: {len(X_test)}")
    
    # Create datasets
    train_dataset = DrugEnzymeDataset(X_train, enzyme_encoding, y_train)
    val_dataset = DrugEnzymeDataset(X_val, enzyme_encoding, y_val)
    test_dataset = DrugEnzymeDataset(X_test, enzyme_encoding, y_test)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    test_loader = DataLoader(test_dataset, batch_size=32)
    
    # Initialize model
    model = DrugEnzymeInteractionModel(
        drug_input_dim=X_smiles.shape[1],  # Morgan fingerprint dimension
        enzyme_input_dim=enzyme_encoding.shape[1],  # Amino acid vocabulary size
        enzyme_seq_length=enzyme_encoding.shape[0],
        hidden_dim=512,
        dropout_rate=0.3
    )
    
    # Print model summary
    print("\nModel architecture:")
    print(model)
    print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    
    # Train model
    print("\nStarting model training...")
    trained_model, history = train_model(
        model, train_loader, val_loader, criterion, optimizer, 
        num_epochs=50, device=device
    )
    
    # Evaluate model
    print("\nEvaluating model on test set...")
    evaluation_results = evaluate_model(trained_model, test_loader, device=device)
    print(f"Test MSE: {evaluation_results['mse']:.4f}")
    print(f"Test RMSE: {evaluation_results['rmse']:.4f}")
    print(f"Test R²: {evaluation_results['r2']:.4f}")
    
    # Plot results
    plot_results(history, evaluation_results)
    
    # Save model
    torch.save(trained_model.state_dict(), 'output/drug_enzyme_model.pt')
    print("Model saved to 'output/drug_enzyme_model.pt'")

except Exception as e:
    print(f"Error in training pipeline: {e}")
    import traceback
    traceback.print_exc()
