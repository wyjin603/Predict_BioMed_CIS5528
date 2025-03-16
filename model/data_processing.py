import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Draw
import torch
from torch.utils.data import Dataset, DataLoader
import requests
import matplotlib.pyplot as plt
import os

class DrugEnzymeDataProcessor:
    def __init__(self, panel_file, potency_file, enzyme_target='p450-cyp3a4'):
        """
        Initialize the data processor for drug-enzyme interaction prediction.
        
        Args:
            panel_file: Path to the CSV file containing drug-enzyme pairs
            potency_file: Path to the CSV file containing potency scores
            enzyme_target: Target enzyme to focus on (default: 'p450-cyp3a4')
        """
        self.panel_file = panel_file
        self.potency_file = potency_file
        self.enzyme_target = enzyme_target
        
        # Create output directory for visualizations
        os.makedirs('output', exist_ok=True)
        
        # Fetch CYP3A4 sequence from UniProt
        try:
            response = requests.get("https://www.uniprot.org/uniprot/P08684.fasta")
            fasta_text = response.text
            # Skip the header line (starts with ">") and join remaining lines
            self.enzyme_sequence = ''.join(fasta_text.split('\n')[1:])
            print(f"Successfully fetched enzyme sequence, length: {len(self.enzyme_sequence)}")
        except Exception as e:
            print(f"Error fetching enzyme sequence: {e}")
            # Fallback sequence if API call fails
            self.enzyme_sequence = "MALIPDLAMETWLLLAVSLVLLYLYGTHSHGLFKKLGIPGPTPLPFLGNILSYHKGFCMFDMECHKKYGKVWGFYDGQQPVLAITDPDMIKTVLVKECYSVFTNRRPFGPVGFMKSAISIAEDEEWKRLRSLLSPTFTSGKLKEMVPIIAQYGDVLVRNLRREAETGKPVTLKDVFGAYSMDVITSTSFGVNIDSLNNPQDPFVENTKKLLRFDFLDPFFLSITVFPFLIPILEVLNICVFPREVTNFLRKSVKRMKESRLEDTQKHRVDFLQLMIDSQNSKETESHKALSDLELVAQSIIFIFAGYETTSSVLSFIMYELATHPDVQQKLQEEIDAVLPNKAPPTYDTVLQMEYLDMVVNETLRLFPIAMRLERVCKKDVEINGMFIPKGVVVMIPSYALHRDPKYWTEPEKFLPERFSKKNKDNIDPYIYTPFGSGPRNCIGMRFALMNMKLALIRVLQNFSFKPCKETQIPLKLSLGGLLQPEKPVVLKVESRDGTVSGA"
            print(f"Using fallback enzyme sequence, length: {len(self.enzyme_sequence)}")

    def load_data(self):
        """Load and preprocess the data files."""
        try:
            # Load panel data
            panel_df = pd.read_csv(self.panel_file)
            
            # Check if columns need to be renamed
            if panel_df.columns[0] != 'index':
                panel_df.columns = ['index', 'smiles', 'enzyme']
            
            # Force convert types
            panel_df['index'] = panel_df['index'].astype(int)
            panel_df['smiles'] = panel_df['smiles'].astype(str)
            panel_df['enzyme'] = panel_df['enzyme'].astype(str)
            
            # Load potency data
            potency_df = pd.read_csv(self.potency_file)
            
            # Check if columns need to be renamed
            if potency_df.columns[0] != 'index':
                potency_df.columns = ['index', 'smiles', 'potency']
            
            # Force convert types
            potency_df['index'] = potency_df['index'].astype(int)
            potency_df['smiles'] = potency_df['smiles'].astype(str)
            potency_df['potency'] = pd.to_numeric(potency_df['potency'], errors='coerce')
            
            # Print data info for debugging
            print(f"Panel data shape: {panel_df.shape}")
            print(f"Potency data shape: {potency_df.shape}")
            
            # Filter for target enzyme
            cyp3a4_panel = panel_df[panel_df['enzyme'] == self.enzyme_target]
            print(f"Filtered data for {self.enzyme_target}: {cyp3a4_panel.shape[0]} entries")
            
            # Merge with potency data
            merged_data = pd.merge(cyp3a4_panel, potency_df, on=['index', 'smiles'])
            print(f"Merged data shape: {merged_data.shape}")
            
            # Handle missing and extreme values
            print(f"Data before cleaning: {merged_data.shape[0]} entries")
            
            # Remove rows with missing potency values
            merged_data = merged_data.dropna(subset=['potency'])
            print(f"Data after removing NaN potency: {merged_data.shape[0]} entries")
            
            # Remove extreme potency values (optional, adjust thresholds as needed)
            potency_upper_limit = 1000  # Adjust based on domain knowledge
            merged_data = merged_data[merged_data['potency'] <= potency_upper_limit]
            print(f"Data after removing extreme potency values: {merged_data.shape[0]} entries")
            
            # Display potency statistics
            print("\nPotency statistics after cleaning:")
            print(merged_data['potency'].describe())
            
            # Display a few examples
            print("\nSample data:")
            print(merged_data.head())
            
            # Verify SMILES data type
            print(f"\nSMILES data type check:")
            for i, smiles in enumerate(merged_data['smiles'].head(3)):
                print(f"SMILES {i}: {smiles}, type: {type(smiles)}")
            
            if merged_data.empty:
                print("Warning: No data available after filtering!")
                return pd.DataFrame(columns=['index', 'smiles', 'enzyme', 'potency'])
            
            return merged_data
        except Exception as e:
            print(f"Error in load_data: {e}")
            import traceback
            traceback.print_exc()
            # Return empty DataFrame as fallback
            return pd.DataFrame(columns=['index', 'smiles', 'enzyme', 'potency'])
    
    def smiles_to_fingerprint(self, smiles, radius=2, nBits=2048):
        """Convert SMILES to Morgan fingerprint."""
        try:
            # Ensure smiles is a string
            if not isinstance(smiles, str):
                print(f"Warning: SMILES is not a string: {smiles}, type: {type(smiles)}")
                smiles = str(smiles)
            
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                print(f"Warning: Could not parse SMILES: {smiles}")
                return np.zeros(nBits)
            
            # Use MorganGenerator instead of deprecated GetMorganFingerprintAsBitVect
            try:
                from rdkit.Chem import rdMolDescriptors
                from rdkit.Chem.rdMolDescriptors import MorganGenerator
                
                # Create Morgan fingerprint using the newer API
                fp_gen = MorganGenerator(radius=radius, fpSize=nBits)
                fp = fp_gen.GetFingerprintAsNumPy(mol)
                return fp
            except (ImportError, AttributeError):
                # Fall back to older method if MorganGenerator is not available
                return np.array(AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits))
            
        except Exception as e:
            print(f"Error processing SMILES {smiles}: {e}")
            return np.zeros(nBits)
    
    def visualize_molecules(self, data, num_samples=5):
        """Visualize a few sample molecules."""
        try:
            # Take a few samples
            samples = data.sample(min(num_samples, len(data)))
            
            mols = []
            labels = []
            
            for _, row in samples.iterrows():
                smiles = row['smiles']
                potency = row['potency']
                mol = Chem.MolFromSmiles(smiles)
                
                if mol:
                    mols.append(mol)
                    labels.append(f"Index: {row['index']}\nPotency: {potency:.2f}")
            
            if mols:
                # Create a grid of molecule images
                img = Draw.MolsToGridImage(mols, molsPerRow=3, subImgSize=(200, 200), 
                                          legends=labels, useSVG=False)
                
                # Save the image
                img.save('output/sample_molecules.png')
                print("Molecule visualization saved to 'output/sample_molecules.png'")
                
                # Return the first molecule for additional visualization
                return mols[0] if mols else None
            else:
                print("No valid molecules to visualize")
                return None
        except Exception as e:
            print(f"Error visualizing molecules: {e}")
            return None
    
    def encode_enzyme_sequence(self, sequence=None):
        """
        Encode enzyme sequence using a simple one-hot encoding.
        In a real implementation, you might use more sophisticated methods.
        """
        if sequence is None:
            sequence = self.enzyme_sequence
            
        # Simple amino acid encoding (you should use more sophisticated methods)
        amino_acids = "ACDEFGHIKLMNPQRSTVWY"
        encoding = np.zeros((len(sequence), len(amino_acids)))
        
        for i, aa in enumerate(sequence):
            if aa in amino_acids:
                encoding[i, amino_acids.index(aa)] = 1
                
        return encoding
    
    def prepare_dataset(self):
        """Prepare dataset for model training."""
        try:
            data = self.load_data()
            
            if data.empty:
                print("Error: No data available for processing")
                return np.array([]), np.array([]), np.array([])
            
            # Visualize data distribution
            self.visualize_data(data)
            
            # Process SMILES strings
            print("Converting SMILES to fingerprints...")
            X_smiles = []
            for i, smiles in enumerate(data['smiles']):
                try:
                    # Ensure smiles is a string
                    if not isinstance(smiles, str):
                        print(f"Converting SMILES at index {i} from {type(smiles)} to string")
                        smiles = str(smiles)
                    
                    # Process fingerprint
                    fp = self.smiles_to_fingerprint(smiles)
                    X_smiles.append(fp)
                except Exception as e:
                    print(f"Error processing SMILES at index {i}: {e}")
                    # Add zeros as fallback
                    X_smiles.append(np.zeros(2048))
            
            # Get enzyme encoding (same for all pairs since we're focusing on CYP3A4)
            print("Encoding enzyme sequence...")
            enzyme_encoding = self.encode_enzyme_sequence()
            
            # Get potency values
            y = data['potency'].values
            
            print(f"Dataset prepared: {len(X_smiles)} samples, enzyme encoding shape: {enzyme_encoding.shape}")
            print(f"Potency range: {np.min(y):.2f} to {np.max(y):.2f}")
            
            return np.array(X_smiles), enzyme_encoding, y
        except Exception as e:
            print(f"Error in prepare_dataset: {e}")
            import traceback
            traceback.print_exc()
            return np.array([]), np.array([]), np.array([])
    
    def visualize_data(self, data):
        """Visualize the data distribution and sample molecules."""
        try:
            os.makedirs('output', exist_ok=True)
            
            # Use non-interactive backend to avoid threading issues
            plt.switch_backend('agg')
            
            # Visualize potency distribution
            plt.figure(figsize=(12, 5))
            
            plt.subplot(1, 2, 1)
            plt.hist(data['potency'], bins=30, alpha=0.7)
            plt.xlabel('Potency')
            plt.ylabel('Frequency')
            plt.title('Potency Distribution')
            
            plt.subplot(1, 2, 2)
            plt.boxplot(data['potency'])
            plt.ylabel('Potency')
            plt.title('Potency Boxplot')
            
            plt.tight_layout()
            plt.savefig('output/potency_distribution.png')
            print("Potency distribution saved to 'output/potency_distribution.png'")
            plt.close('all')  # Close all figures to free memory
            
            # Visualize molecules with limited samples to avoid threading issues
            self.visualize_molecules(data, num_samples=3)  # Reduced from 5 to 3
            
        except Exception as e:
            print(f"Error visualizing data: {e}")
            import traceback
            traceback.print_exc()


class DrugEnzymeDataset(Dataset):
    """PyTorch Dataset for drug-enzyme pairs."""
    
    def __init__(self, X_smiles, enzyme_encoding, y):
        self.X_smiles = X_smiles
        self.enzyme_encoding = enzyme_encoding
        self.y = y
        
    def __len__(self):
        return len(self.X_smiles)
    
    def __getitem__(self, idx):
        return {
            'drug': torch.FloatTensor(self.X_smiles[idx]),
            'enzyme': torch.FloatTensor(self.enzyme_encoding),
            'potency': torch.FloatTensor([self.y[idx]])
        } 