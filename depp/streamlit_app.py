import datetime
import random
import math

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from tape.tokenizers import TAPETokenizer
import deepchem as dc
from transformers import AutoTokenizer

from rdkit import Chem
from rdkit.Chem import AllChem
import pickle
import os
from rdkit import DataStructs

import torch
import torch.nn as nn
import torch.nn.functional as F

# --- PyTorch Model Definition ---
class DrugEnzymeInteractionModel(nn.Module):
    def __init__(self, drug_input_dim=2048, enzyme_input_dim=20, enzyme_seq_length=500,
                 hidden_dim=512, dropout_rate=0.2):
        """
        Model for predicting drug-enzyme interaction potency. (Using provided class)
        """
        super(DrugEnzymeInteractionModel, self).__init__()

<<<<<<< codespace-crispy-happiness-wq9rjr4jpj6cpx
        # Drug processing branch
        self.drug_fc1 = nn.Linear(drug_input_dim, hidden_dim)
        self.drug_bn1 = nn.BatchNorm1d(hidden_dim)
        self.drug_fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.drug_bn2 = nn.BatchNorm1d(hidden_dim)

        # Enzyme processing branch
        # Assuming enzyme_input_dim is the number of channels (vocab size)
        self.enzyme_conv = nn.Conv1d(enzyme_input_dim, 64, kernel_size=3, padding=1)
        self.enzyme_pool = nn.MaxPool1d(2)
        # Calculate the sequence length after conv and pool for LSTM input
        # Conv1d with padding=1 preserves length. MaxPool1d(2) halves it.
        lstm_input_size = 64 # Output channels of Conv1d
        # Assuming sequence length needs careful handling if pooling changes it significantly
        # For now, assuming LSTM handles variable lengths or input is padded appropriately
        self.enzyme_lstm = nn.LSTM(lstm_input_size, hidden_dim//2, batch_first=True, bidirectional=True)

        # Attention mechanism for enzyme sequence
        self.attention = nn.Linear(hidden_dim, 1) # LSTM output is hidden_dim (hidden_dim//2 * 2 for bidirectional)

        # Integration layers
        self.integration_fc = nn.Linear(hidden_dim * 2, hidden_dim) # drug (hidden_dim) + enzyme (hidden_dim)
        self.integration_bn = nn.BatchNorm1d(hidden_dim)

        # Prediction layers
        self.pred_fc1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.pred_bn1 = nn.BatchNorm1d(hidden_dim // 2)
        self.pred_fc2 = nn.Linear(hidden_dim // 2, 1)

        # Dropout
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, drug, enzyme):
        """
        Forward pass through the model.

        Args:
            drug: Drug fingerprint tensor [batch_size, drug_input_dim]
            enzyme: Enzyme sequence encoding [batch_size, enzyme_seq_length, enzyme_input_dim]

        Returns:
            Predicted potency score [batch_size, 1]
        """
        # Process drug
        # Need to check batchnorm applicability if batch_size is 1 during inference
        # Using eval() mode should handle this correctly by using running stats
        drug = F.relu(self.drug_bn1(self.drug_fc1(drug)))
        drug = self.dropout(drug)
        drug = F.relu(self.drug_bn2(self.drug_fc2(drug)))
        drug = self.dropout(drug)

        # Process enzyme
        # Reshape for Conv1d: [batch, channels (enzyme_input_dim), length (enzyme_seq_length)]
        enzyme = enzyme.permute(0, 2, 1)
        enzyme = F.relu(self.enzyme_conv(enzyme))
        enzyme = self.enzyme_pool(enzyme) # Shape becomes [batch, 64, length/2]

        # Reshape for LSTM: [batch, length, channels]
        enzyme = enzyme.permute(0, 2, 1) # Shape becomes [batch, length/2, 64]
        enzyme, _ = self.enzyme_lstm(enzyme) # Shape becomes [batch, length/2, hidden_dim]

        # Attention mechanism
        # Apply attention linear layer to each time step output
        attention_weights = F.softmax(self.attention(enzyme), dim=1) # Shape [batch, length/2, 1]
        # Weighted sum: element-wise multiply weights with LSTM outputs and sum across time steps
        enzyme = torch.sum(attention_weights * enzyme, dim=1) # Shape becomes [batch, hidden_dim]

        # Integrate drug and enzyme features
        combined = torch.cat([drug, enzyme], dim=1) # Shape [batch, hidden_dim * 2]
        combined = F.relu(self.integration_bn(self.integration_fc(combined)))
        combined = self.dropout(combined)

        # Prediction
        output = F.relu(self.pred_bn1(self.pred_fc1(combined)))
        output = self.dropout(output)
        output = self.pred_fc2(output) # Shape [batch, 1]

        return output
=======
class ImageVAE(nn.Module):
    def __init__(self, latent_dim=128):
        super(ImageVAE, self).__init__()
        
        # Encoder for grayscale images (input: 1x64x64)
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 4, stride=2, padding=1),   # (32,32,32)
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),      # (64,16,16)
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),     # (128,8,8)
            nn.ReLU(),
            nn.Conv2d(128, 256, 4, stride=2, padding=1),    # (256,4,4)
            nn.ReLU(),
        )

        self.fc_mu = nn.Linear(256*4*4, latent_dim)
        self.fc_logvar = nn.Linear(256*4*4, latent_dim)

        # Decoder
        self.decoder_input = nn.Linear(latent_dim, 256*4*4)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),  # (128,8,8)
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),  # (64,16,16)
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),  # (32,32,32)
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 4, stride=2, padding=1),  # (1,64,64)
            nn.Sigmoid(),
        )

    def encode(self, x):
        enc = self.encoder(x)
        enc = enc.view(-1, 256 * 4 * 4)
        return self.fc_mu(enc), self.fc_logvar(enc), enc

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        dec_input = self.decoder_input(z)
        dec_input = dec_input.view(-1, 256, 4, 4)
        return self.decoder(dec_input)

    def forward(self, x):
        mu, logvar, enc = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar, enc


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=256):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # [1, max_len, d_model]

    def forward(self, x):
        # x: [B, T, D]
        return x + self.pe[:, : x.size(1)]


class SmilesTransformerVAE(nn.Module):
    def __init__(
        self,
        vocab_size,
        d_model=256,
        nhead=8,
        num_enc_layers=3,
        num_dec_layers=3,
        latent_dim=128,
        max_length=100,
    ):
        super().__init__()
        self.d_model = d_model
        self.max_length = max_length

        # token embedding + positional encoding
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_encoder = PositionalEncoding(d_model, max_len=max_length)

        # Transformer Encoder
        enc_layer = nn.TransformerEncoderLayer(d_model, nhead,
                                               dim_feedforward=512)
        self.encoder = nn.TransformerEncoder(enc_layer, num_enc_layers)

        # VAE bottleneck
        self.fc_mu = nn.Linear(d_model, latent_dim)
        self.fc_logvar = nn.Linear(d_model, latent_dim)
        self.fc_z2mem = nn.Linear(latent_dim, d_model)

        # Transformer Decoder
        dec_layer = nn.TransformerDecoderLayer(d_model, nhead,
                                               dim_feedforward=512)
        self.decoder = nn.TransformerDecoder(dec_layer, num_dec_layers)

        self.output_fc = nn.Linear(d_model, vocab_size)

    def encode(self, src, src_key_padding_mask):
        # src: [B, T]
        x = self.embedding(src) * math.sqrt(self.d_model)  # [B, T, D]
        x = self.pos_encoder(x)
        # Transformer expects [T, B, D]
        enc_out = self.encoder(
            x.transpose(0, 1),
            src_key_padding_mask=src_key_padding_mask
        )  # [T, B, D]
        # use the encoding at position 0 (i.e. the <CLS> token) as summary
        cls_rep = enc_out[0]  # [B, D]
        mu = self.fc_mu(cls_rep)     # [B, L]
        logvar = self.fc_logvar(cls_rep)  # [B, L]
        return mu, logvar, enc_out

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std  # [B, L]

    def decode(self, z, y, src_key_padding_mask, teacher_forcing_ratio=1.0):
        B, T = y.size()
        device = y.device

        tgt = y[:, :-1]                             # [B, T−1]
        tgt_emb = self.embedding(tgt) * math.sqrt(self.d_model)
        tgt_emb = self.pos_encoder(tgt_emb)         # [B, T−1, D]

        mem = self.fc_z2mem(z).unsqueeze(0)          # [1, B, D]

        tgt_mask = nn.Transformer.generate_square_subsequent_mask(
            T-1).to(device)
        tgt_mask = tgt_mask.to(torch.bool)
        #  padding mask for target: [B, T−1]
        tgt_key_padding_mask = src_key_padding_mask[:, :-1]

        dec_out = self.decoder(
            tgt_emb.transpose(0, 1),
            mem,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask
        )  # [T−1, B, D]

        # project to vocab and return [B, T−1, V]
        logits = self.output_fc(dec_out.transpose(0, 1))
        return logits

    def forward(self, src, src_key_padding_mask, teacher_forcing_ratio=1.0):

        mu, logvar, enc = self.encode(src, src_key_padding_mask)
        z = self.reparameterize(mu, logvar)

        logits = self.decode(z, src, src_key_padding_mask,
                             teacher_forcing_ratio)
        return logits, mu, logvar, enc


class EnzymeTransformerVAE(nn.Module):
    def __init__(
        self,
        vocab_size,
        d_model=256,
        nhead=8,
        num_enc_layers=3,
        num_dec_layers=3,
        latent_dim=128,
        max_length=100,
    ):
        super().__init__()
        self.d_model = d_model
        self.max_length = max_length

        # token embedding + positional encoding
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_encoder = PositionalEncoding(d_model, max_len=max_length)

        # Transformer Encoder
        enc_layer = nn.TransformerEncoderLayer(d_model, nhead,
                                               dim_feedforward=512)
        self.encoder = nn.TransformerEncoder(enc_layer, num_enc_layers)

        # VAE bottleneck
        self.fc_mu = nn.Linear(d_model, latent_dim)
        self.fc_logvar = nn.Linear(d_model, latent_dim)
        self.fc_z2mem = nn.Linear(latent_dim, d_model)

        # Transformer Decoder
        dec_layer = nn.TransformerDecoderLayer(d_model, nhead,
                                               dim_feedforward=512)
        self.decoder = nn.TransformerDecoder(dec_layer, num_dec_layers)

        self.output_fc = nn.Linear(d_model, vocab_size)

    def encode(self, src, src_key_padding_mask):
        # src: [B, T]
        x = self.embedding(src) * math.sqrt(self.d_model)  # [B, T, D]
        x = self.pos_encoder(x)
        # Transformer expects [T, B, D]
        enc_out = self.encoder(
            x.transpose(0, 1),
            src_key_padding_mask=src_key_padding_mask
        )  # [T, B, D]
        # use the encoding at position 0 (i.e. the <CLS> token) as summary
        cls_rep = enc_out[0]  # [B, D]
        mu = self.fc_mu(cls_rep)     # [B, L]
        logvar = self.fc_logvar(cls_rep)  # [B, L]
        return mu, logvar, enc_out

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std  # [B, L]

    def decode(self, z, y, src_key_padding_mask, teacher_forcing_ratio=1.0):

        B, T = y.size()
        device = y.device

        tgt = y[:, :-1]                             # [B, T−1]
        tgt_emb = self.embedding(tgt) * math.sqrt(self.d_model)
        tgt_emb = self.pos_encoder(tgt_emb)         # [B, T−1, D]

        mem = self.fc_z2mem(z).unsqueeze(0)          # [1, B, D]

        tgt_mask = nn.Transformer.generate_square_subsequent_mask(
            T-1).to(device)
        tgt_mask = tgt_mask.to(torch.bool)
        # padding mask for target: [B, T−1]
        tgt_key_padding_mask = src_key_padding_mask[:, :-1]

        dec_out = self.decoder(
            tgt_emb.transpose(0, 1),
            mem,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask
        )  # [T−1, B, D]

        # project to vocab and return [B, T−1, V]
        logits = self.output_fc(dec_out.transpose(0, 1))
        return logits

    def forward(self, src, src_key_padding_mask, teacher_forcing_ratio=1.0):

        mu, logvar, enc = self.encode(src, src_key_padding_mask)
        z = self.reparameterize(mu, logvar)

        logits = self.decode(z, src, src_key_padding_mask,
                             teacher_forcing_ratio)
        return logits, mu, logvar, enc


class PotencyPredictor(nn.Module):
    def __init__(self, image_vae_model, smile_transf_model,
                 enzyme_transf_model, img_latent_dim=128,
                 smile_latent_dim=128, enzyme_latent_dim=128):
        super().__init__()
        self.img_encoder = image_vae_model
        self.smile_encoder = smile_transf_model
        self.enzyme_encoder = enzyme_transf_model
        self.total_dim = img_latent_dim + smile_latent_dim + enzyme_latent_dim
        
        self.predictor = nn.Sequential(
            nn.Linear(self.total_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )
    
    def forward(self, img, smile_ids, enzyme_ids):
        mu_img, _, img_enc = self.img_encoder.encode(img)
        smiles_mask = smile_ids == smiles_tokenizer.pad_token_id
        mu_smile, _, smile_enc = self.smile_encoder.encode(smile_ids,
                                                           smiles_mask)
        enzyme_mask = enzyme_ids == 0
        mu_enzyme, _, enzyme_enc = self.enzyme_encoder.encode(enzyme_ids,
                                                              enzyme_mask)
        
        x = torch.cat([mu_img, mu_smile, mu_enzyme], dim=1)
        encs = torch.cat([img_enc, smile_enc, enzyme_enc], dim=0)
        return self.predictor(x).squeeze(1)
>>>>>>> main

# --- Configuration ---
MODEL_PARAMS = {
    "drug_input_dim": 2048,
    "enzyme_input_dim": 20, # Matches the number of standard amino acids
    "enzyme_seq_length": 500, # Max sequence length model was trained on
    "hidden_dim": 512,
    "dropout_rate": 0.2 # Set to 0 for eval usually, but model definition includes it
}
MODEL_FILENAME = "depp_model.pth" # Your PyTorch state dictionary file

# Update other configs based on MODEL_PARAMS
FP_LENGTH = MODEL_PARAMS["drug_input_dim"]
MAX_SEQ_LENGTH = MODEL_PARAMS["enzyme_seq_length"]
AMINO_ACIDS = 'ACDEFGHIKLMNPQRSTVWY' # Standard 20 amino acids
# VOCAB_SIZE must match enzyme_input_dim for one-hot encoding
VOCAB_SIZE = MODEL_PARAMS["enzyme_input_dim"]
if len(AMINO_ACIDS) != VOCAB_SIZE:
    st.warning(f"Mismatch: AMINO_ACIDS length ({len(AMINO_ACIDS)}) != enzyme_input_dim ({VOCAB_SIZE}). Check configuration.")

CHAR_TO_INT = {char: i for i, char in enumerate(AMINO_ACIDS)}

# Determine device
#DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#st.info(f"Using device: {DEVICE}")
# --- End Configuration ---


# --- Helper Functions ---
@st.cache_resource # Cache the loaded PyTorch model
def load_pytorch_model(model_class, model_params, filename):
    """Loads the PyTorch model state dictionary."""
    if not os.path.exists(filename):
        st.error(f"Error: Model file '{filename}' not found.")
        st.stop()
    try:
        # Instantiate model with specified parameters
        model = model_class(**model_params)
        # Load the saved state dictionary
        # Use map_location to load correctly regardless of where it was saved
        model.load_state_dict(torch.load(filename))
        # Set the model to evaluation mode (important!)
        #model.eval()
        # Move model to the designated device
        #model.to(device)
        #st.success(f"PyTorch model '{filename}' loaded successfully to {device}.")
        return model
    except Exception as e:
        st.error(f"Error loading PyTorch model '{filename}': {e}")
        st.exception(e) # Show detailed traceback
        st.stop()

# smiles_to_fingerprint remains the same, returning a NumPy array
@st.cache_data
def smiles_to_fingerprint(smiles, fp_length):
    """Converts SMILES string to Morgan fingerprint (numpy array)."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            st.warning(f"Could not parse SMILES: '{smiles}'. Please enter a valid SMILES string.")
            return None
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=fp_length)
        arr = np.zeros((1,), dtype=np.int8)
        DataStructs.ConvertToNumpyArray(fp, arr)
        return arr.astype(float) # Return numpy array
    except Exception as e:
        st.error(f"Error generating fingerprint for SMILES '{smiles}': {e}")
        return None

# sequence_to_one_hot needs adjustment based on VOCAB_SIZE matching len(AMINO_ACIDS)
@st.cache_data
def sequence_to_one_hot(sequence, char_to_int_map, max_len, vocab_size):
    """
    Converts amino acid sequence to one-hot encoded numpy array.
    Handles padding/truncation. Errors on unknown characters if vocab_size == len(amino_acids).
    """
    sequence = sequence.upper() # Ensure consistent case

    # Pad or truncate sequence
    if len(sequence) > max_len:
        processed_sequence = sequence[:max_len]
    else:
        # Use a padding character NOT in AMINO_ACIDS if needed, but model might not expect it.
        # Let's assume padding isn't explicitly encoded if vocab_size matches AMINO_ACIDS count.
        # Sequences shorter than max_len might need careful handling depending on model training.
        # Common approach: pad with zeros implicitly by not setting any '1'.
        processed_sequence = sequence
        # Alternatively, pad with a specific character if model expects it:
        # processed_sequence = sequence + 'X' * (max_len - len(sequence))

    one_hot = np.zeros((max_len, vocab_size), dtype=np.float32)

    for i, char in enumerate(processed_sequence):
        if char in char_to_int_map:
            one_hot[i, char_to_int_map[char]] = 1.0
        else:
            # If character is not a known amino acid
            st.error(f"Unknown character '{char}' found in sequence at position {i}. "
                     f"Expected characters: {list(char_to_int_map.keys())}. Cannot proceed.")
            return None # Stop processing if unknown char found and no 'unknown' category

    # If sequence was shorter than max_len, the remaining rows in one_hot will be all zeros.
    # This implicitly handles padding if the model was trained this way (e.g., using masking later).

    return one_hot # Return numpy array


def apply_mutation(sequence, position, new_amino_acid):
    """Apply a point mutation to a protein sequence"""
    # Convert to 0-based index
    pos_idx = position - 1
   
    # Verify the sequence is long enough
    if pos_idx >= len(sequence):
        raise ValueError(
            f"Position {position} exceeds sequence length {len(sequence)}")
   
    # Create new sequence with mutation
    if (new_amino_acid == 'STOP'):
        new_sequence = sequence[:pos_idx]
    else:
        new_sequence = sequence[:pos_idx] + new_amino_acid + sequence[
            pos_idx+1:]
    return new_sequence
 

def get_cyp3a4_variant_sequence(variant_number, wild_type_sequence):
    """Generate a CYP3A4 variant sequence based on variant number"""
    # Mapping of variant numbers to mutations
    variant_mutations = {
        1: [{"position": -1, "new_aa": ""}],  # Wild-type
        2: [{"position": 222, "new_aa": "P"}],  # S222P
        3: [{"position": 445, "new_aa": "T"}],  # M445T
        4: [{"position": 118, "new_aa": "V"}],  # I118V
        5: [{"position": 218, "new_aa": "R"}],  # P218R
        6: [{"position": 277, "new_aa": ""}],    # 277 frameshift
        7: [{"position": 56, "new_aa": "D"}],     # G56D
        8: [{"position": 130, "new_aa": "Q"}],    # R130Q
        9: [{"position": 170, "new_aa": "I"}],  # V170I
        10: [{"position": 174, "new_aa": "H"}],  # D174H
        11: [{"position": 363, "new_aa": "M"}],  # T363M
        12: [{"position": 373, "new_aa": "F"}],  # L373F
        13: [{"position": 416, "new_aa": "L"}],  # P416L
        14: [{"position": 15, "new_aa": "P"}],  # L15P
        15: [{"position": 162, "new_aa": "Q"}],  # R162Q
        16: [{"position": 185, "new_aa": "S"}],  # T185S
        17: [{"position": 189, "new_aa": "S"}],  # F189S
        18: [{"position": 293, "new_aa": "P"}],  # L293P
        19: [{"position": 467, "new_aa": "S"}],  # P467S
        20: [{"position": 488, "new_aa": ""}],  # 488 frameshift
        21: [{"position": 319, "new_aa": "C"}],  # Y319C
        22: [{"position": -1, "new_aa": ""}],  # Intronic Variance
        23: [{"position": 162, "new_aa": "W"}],  # R162W
        24: [{"position": 200, "new_aa": "H"}],  # Q200H
        25: [{"position": 324, "new_aa": "Q"}],  # H324Q
        26: [{"position": 268, "new_aa": "STOP"}],  # R268STOP
        27: [{"position": 22, "new_aa": "V"}],  # L22V
        28: [{"position": 22, "new_aa": "V"}],  # L22V
        29: [{"position": 113, "new_aa": "I"}],  # F113I
        30: [{"position": 130, "new_aa": "STOP"}],  # R130STOP
        31: [{"position": 324, "new_aa": "Q"}],  # H324Q
        32: [{"position": 335, "new_aa": "T"}],  # I335T
        33: [{"position": 370, "new_aa": "S"}],  # A370S
        34: [{"position": 427, "new_aa": "V"}]   # I427V
    }
   
    # If variant not in mapping, return wild-type
    if variant_number not in variant_mutations:
        print(f"Warning: CYP3A4*{variant_number} not defined in mapping")
        return wild_type_sequence
   
    # For wild-type, return as is
    if variant_number == 1:
        return wild_type_sequence
   
    # Apply each mutation in the variant
    sequence = wild_type_sequence
    for mutation in variant_mutations[variant_number]:
        sequence = apply_mutation(sequence, mutation["position"],
                                  mutation["new_aa"])
   
    return sequence

# --- Load Model ---
# Use the new function to load the PyTorch model
<<<<<<< codespace-crispy-happiness-wq9rjr4jpj6cpx
#model = load_pytorch_model(DrugEnzymeInteractionModel, MODEL_PARAMS, MODEL_FILENAME)
=======
# model = load_pytorch_model(DrugEnzymeInteractionModel, MODEL_PARAMS, MODEL_FILENAME, DEVICE)


tape_tokenizer = TAPETokenizer(vocab="unirep")  

smiles_tokenizer = AutoTokenizer.from_pretrained(
    "seyonec/ChemBERTa-zinc-base-v1")

smile_vocab = smiles_tokenizer.get_vocab()
# enzyme_vocab = tape_tokenizer.get_vocab()

smile_vocab_size = len(smile_vocab)
enzyme_vocab_size = tape_tokenizer.vocab_size

image_model = ImageVAE()
smile_transf_model = SmilesTransformerVAE(
    vocab_size=len(smiles_tokenizer), d_model=256, nhead=8,
    num_enc_layers=3, num_dec_layers=3,
    latent_dim=128, max_length=256
)
enzyme_transf_model = EnzymeTransformerVAE(
    vocab_size=enzyme_vocab_size, d_model=256, nhead=8,
    num_enc_layers=3, num_dec_layers=3,
    latent_dim=128, max_length=100
)

model = PotencyPredictor(image_model, smile_transf_model, enzyme_transf_model)

image_model.load_state_dict(torch.load('./models/image_vae.pth',
                                       weights_only=True))
smile_transf_model.load_state_dict(torch.load('./models/smile_transf_vae.pth',
                                              weights_only=True))
enzyme_transf_model.load_state_dict(torch.load(
    './models/enzyme_transf_model.pth', weights_only=True))
model.load_state_dict(torch.load('./models/potency_model.pth',
                                 weights_only=True))
>>>>>>> main

# --- Streamlit App Layout ---
# (Keep the layout parts: title, description, example data, form)
st.set_page_config(page_title="Drug-Enzyme Potency Predictor (DEPP)")
st.title("🧪 Drug-Enzyme Potency Predictor (DEPP)")
st.write(
    """
    Enter a drug's SMILES string and an enzyme's amino acid sequence to predict
    their interaction potency score using a deep learning model.
    """
)
st.markdown("**Disclaimer:** This tool provides predictions based on a specific model. \
            These predictions are for informational purposes only and should not be \
            considered a substitute for experimental validation or expert advice.")
st.divider()

# --- Example Data (Optional but helpful) ---
if "df" not in st.session_state:
    example_smiles = [
        "CCCC(=O)NC1=CC(=C(C=C1)N2CCN(CC2)CC)Cl.Cl",
        "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",
        "CC(=O)OC1=CC=CC=C1C(=O)O"
    ]
    example_sequences = [
        "MAEGEITTFTALTEKFNLPPGNYKKPKLLYCSNGGHFLRILPDGTVDGTRDRSDQHIQLQLSAESVGEVYIKSTETGQYLAMDTSGLLYGSQTPSEECLFLERLEENHYNTYTSKKHAKNWFVGLKKNGSCKRGPRTHYGQKAILFLPLPV",
        # Make sure sequence length is reasonable compared to MAX_SEQ_LENGTH
        "MGHHHHHHSSGLVPRGSHMAEGEITTFTALTEKFNLPPGNYKKPKLLYCSNGGHFLRILPDGTVDGTRDRSDQHIQLQLSAESVGEVYIKSTETGQYLAMDTSGLLYGSQTPSEECLFLERLEENHYNTYTSKKHAKNWFVGLKKNGSCKRGPRTHYGQKAILFLPLPV",
    ] * 3 # Repeat sequences to fill 5 examples
    data = {
        "ID": [f"EXAMPLE-{i}" for i in range(1, 6)],
        "SMILES": random.choices(example_smiles, k=5),
        "Amino Acid Sequence": random.choices(example_sequences, k=5),
        "Predicted Potency Score": np.random.rand(5) * 10, # Placeholder scores
        "Date Submitted": [
            datetime.date(2024, 1, 1) + datetime.timedelta(days=random.randint(0, 100))
            for _ in range(5)
        ],
    }
    st.session_state.df = pd.DataFrame(data)


# --- Input Form ---
st.header("Enter Data for Prediction")
with st.form("prediction_form"):
    smiles_input = st.text_area("Enter Drug SMILES String", height=100, placeholder="e.g., CC(=O)OC1=CC=CC=C1C(=O)O")
    enzyme_sequence_input = st.text_area("Enter Enzyme Amino Acid Sequence", height=150, placeholder="e.g., MAEGEITTFT...")

    submitted = st.form_submit_button("Predict Potency")

# --- Prediction Logic & Output ---
if submitted and model: # Only proceed if submitted and model is loaded
    st.subheader("Prediction Result")

    if not smiles_input:
        st.warning("Please enter a SMILES string.")
    elif not enzyme_sequence_input:
        st.warning("Please enter an enzyme sequence.")
    else:
        # 1. Process Inputs (Get NumPy arrays)
        drug_fp_np = smiles_to_fingerprint(smiles_input, FP_LENGTH)
        enzyme_ohe_np = sequence_to_one_hot(
            enzyme_sequence_input,
            CHAR_TO_INT,
            MAX_SEQ_LENGTH,
            VOCAB_SIZE
        )

        if drug_fp_np is None or enzyme_ohe_np is None:
            st.error("Input processing failed. Please check the SMILES string and sequence (ensure only valid amino acids are present).")
        else:
            # 2. Prepare data for the PyTorch model
            try:
                # Convert NumPy arrays to PyTorch tensors
                # Add batch dimension (unsqueeze(0)) -> [1, ...]
                # Move tensors to the correct device
                drug_tensor = torch.tensor(drug_fp_np, dtype=torch.float32).unsqueeze(0).to(DEVICE)
                enzyme_tensor = torch.tensor(enzyme_ohe_np, dtype=torch.float32).unsqueeze(0).to(DEVICE)

                # Verify tensor shapes (optional but good for debugging)
                # st.write("Drug Tensor Shape:", drug_tensor.shape) # Should be [1, drug_input_dim]
                # st.write("Enzyme Tensor Shape:", enzyme_tensor.shape) # Should be [1, enzyme_seq_length, enzyme_input_dim]

                # 3. Make Prediction with PyTorch model
                with st.spinner("Predicting..."), torch.no_grad(): # Use no_grad context for inference
                    prediction = model(drug_tensor, enzyme_tensor)

                # Extract scalar value from tensor (output shape is likely [1, 1])
                potency_score = prediction.item()

                st.success(f"Predicted Potency Score: **{potency_score:.4f}**")

                # 4. Add result to the session dataframe
                recent_ticket_number = len(st.session_state.df) + 1
                today = datetime.datetime.now().date()
                df_new = pd.DataFrame(
                    [
                        {
                            "ID": f"PRED-{recent_ticket_number}",
                            "SMILES": smiles_input,
                            "Amino Acid Sequence": enzyme_sequence_input,
                            "Predicted Potency Score": potency_score,
                            "Date Submitted": today
                        }
                    ]
                )
                st.session_state.df = pd.concat([df_new, st.session_state.df], ignore_index=True)

            except Exception as e:
                st.error(f"An unexpected error occurred during PyTorch prediction: {e}")
                st.exception(e) # Shows detailed traceback for debugging

st.divider()

# --- Display Results Table ---
st.header("Prediction History & Examples")
st.write(f"Total entries: `{len(st.session_state.df)}`")
st.info("Displaying recent predictions and examples. You can sort the table by clicking column headers.")

# (Keep the dataframe display part, it should work fine with the updated df)
st.dataframe(
    st.session_state.df,
    use_container_width=True,
    hide_index=True,
    column_config={
        "SMILES": st.column_config.TextColumn("SMILES", width="medium", help="Drug SMILES String"),
        "Amino Acid Sequence": st.column_config.TextColumn("Sequence", width="large", help="Enzyme Amino Acid Sequence"),
        "Predicted Potency Score": st.column_config.NumberColumn(
            "Potency Score",
            help="Predicted potency score (model output)",
            format="%.4f",
        ),
         "Date Submitted": st.column_config.DateColumn(
            "Date",
            format="YYYY-MM-DD",
        )
    },
)

# --- Basic Statistics Chart ---
# (Keep the statistics chart part, it should work fine)
st.header("Statistics")
st.write("##### Distribution of Predicted Potency Scores")

if not st.session_state.df.empty and "Predicted Potency Score" in st.session_state.df.columns:
     # Ensure scores are numeric before plotting
    plot_df = st.session_state.df.copy()
    plot_df["Predicted Potency Score"] = pd.to_numeric(plot_df["Predicted Potency Score"], errors='coerce')
    plot_df.dropna(subset=["Predicted Potency Score"], inplace=True)

    if not plot_df.empty:
        potency_chart = alt.Chart(plot_df).mark_bar().encode(
            alt.X("Predicted Potency Score", bin=alt.Bin(maxbins=20), title="Predicted Potency Score"),
            alt.Y('count()', title='Number of Entries'),
            tooltip=[alt.Tooltip("Predicted Potency Score", bin=True), 'count()']
        ).properties(
            title='Distribution of Predicted Potency Scores'
        )
        st.altair_chart(potency_chart, use_container_width=True, theme="streamlit")
    else:
        st.write("No valid numeric prediction data available for statistics.")
else:
    st.write("No prediction data available to generate statistics.")
