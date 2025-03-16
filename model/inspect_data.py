import pandas as pd
import os

def inspect_csv_file(file_path):
    """Inspect a CSV file to understand its structure."""
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return
    
    print(f"\nInspecting file: {file_path}")
    
    try:
        # Read first few lines directly
        with open(file_path, 'r') as f:
            print("First 5 lines of raw file:")
            for i, line in enumerate(f):
                if i < 5:
                    print(f"  Line {i+1}: {line.strip()}")
                else:
                    break
        
        # Try reading with pandas
        df = pd.read_csv(file_path)
        print(f"\nFile Statistics:")
        print(f"Shape: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")
        print(f"Data types:\n{df.dtypes}")
        print("\nMissing values:\n{df.isnull().sum()}")
        
        if 'potency' in df.columns:
            print("\nPotency statistics:")
            print(df['potency'].describe())
        
        print("\nFirst 3 rows:")
        print(df.head(3))
        
        # Check for any invalid or problematic values
        if 'smiles' in df.columns:
            print("\nSMILES string examples:")
            print(df['smiles'].head())
            print("\nUnique SMILES count:", df['smiles'].nunique())
        
    except Exception as e:
        print(f"Error inspecting file: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Inspect the data files
    inspect_csv_file("model_data/smiles_panel.csv")
    inspect_csv_file("model_data/smiles_potency.csv") 