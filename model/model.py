import torch
import torch.nn as nn
import torch.nn.functional as F

class DrugEnzymeInteractionModel(nn.Module):
    def __init__(self, drug_input_dim=2048, enzyme_input_dim=20, enzyme_seq_length=500, 
                 hidden_dim=512, dropout_rate=0.2):
        """
        Model for predicting drug-enzyme interaction potency.
        
        Args:
            drug_input_dim: Dimension of drug fingerprint
            enzyme_input_dim: Dimension of enzyme encoding (amino acid vocabulary size)
            hidden_dim: Hidden dimension for neural networks
            dropout_rate: Dropout rate for regularization
        """
        super(DrugEnzymeInteractionModel, self).__init__()
        
        # Drug processing branch
        self.drug_fc1 = nn.Linear(drug_input_dim, hidden_dim)
        self.drug_bn1 = nn.BatchNorm1d(hidden_dim)
        self.drug_fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.drug_bn2 = nn.BatchNorm1d(hidden_dim)
        
        # Enzyme processing branch
        self.enzyme_conv = nn.Conv1d(enzyme_input_dim, 64, kernel_size=3, padding=1)
        self.enzyme_pool = nn.MaxPool1d(2)
        self.enzyme_lstm = nn.LSTM(64, hidden_dim//2, batch_first=True, bidirectional=True)
        
        # Attention mechanism for enzyme sequence
        self.attention = nn.Linear(hidden_dim, 1)
        
        # Integration layers
        self.integration_fc = nn.Linear(hidden_dim * 2, hidden_dim)
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
            Predicted potency score
        """
        # Process drug
        drug = F.relu(self.drug_bn1(self.drug_fc1(drug)))
        drug = self.dropout(drug)
        drug = F.relu(self.drug_bn2(self.drug_fc2(drug)))
        drug = self.dropout(drug)
        
        # Process enzyme
        # Reshape for Conv1d: [batch, channels, length]
        batch_size = enzyme.size(0)
        enzyme = enzyme.permute(0, 2, 1)
        enzyme = F.relu(self.enzyme_conv(enzyme))
        enzyme = self.enzyme_pool(enzyme)
        
        # Reshape for LSTM: [batch, length, channels]
        enzyme = enzyme.permute(0, 2, 1)
        enzyme, _ = self.enzyme_lstm(enzyme)
        
        # Attention mechanism
        attention_weights = F.softmax(self.attention(enzyme), dim=1)
        enzyme = torch.sum(attention_weights * enzyme, dim=1)
        
        # Integrate drug and enzyme features
        combined = torch.cat([drug, enzyme], dim=1)
        combined = F.relu(self.integration_bn(self.integration_fc(combined)))
        combined = self.dropout(combined)
        
        # Prediction
        output = F.relu(self.pred_bn1(self.pred_fc1(combined)))
        output = self.dropout(output)
        output = self.pred_fc2(output)
        
        return output
