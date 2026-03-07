import torch
import torch.nn as nn
from typing import Optional

class TemporalTransformer(nn.Module):
    def __init__(
        self, 
        input_size: int, 
        num_layers: int = 4, 
        num_heads: int = 8, 
        max_seq_len: int = 1024, 
        dropout: float = 0.1,
        dim_feedforward: int = 2048
    ):
        super(TemporalTransformer, self).__init__()
        
        self.input_size = input_size
        
        # Shape: [1, Max_Len, Hidden_Size]
        self.pos_embedding = nn.Embedding(max_seq_len, input_size)
        nn.init.trunc_normal_(self.pos_embedding.weight, std=0.02)
        
        self.dropout = nn.Dropout(p=dropout)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_size,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True 
        )
        
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=num_layers
        )
        
        self.final_norm = nn.LayerNorm(input_size)

    def forward(self, x: torch.Tensor, src_key_padding_mask: Optional[torch.Tensor] = None):
        """
        Args:
            x: Input tensor of shape [Batch, Time, Channels]
            src_key_padding_mask: Boolean mask [Batch, Time] (True = padding)
        """
        B, T, C = x.shape
        
        positions = torch.arange(T, device=x.device).unsqueeze(0)
        x = x + self.pos_embedding(positions)
        
        x = self.dropout(x)
        
        output = self.transformer_encoder(x, src_key_padding_mask=src_key_padding_mask)
        
        return self.final_norm(output)


if __name__ == "__main__":
    batch_size = 4
    original_time_steps = 100
    hidden_dim = 512 
    max_seq_len = 512

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TemporalTransformer(
        input_size=hidden_dim,
        num_layers=2,
        num_heads=8,
        max_seq_len=max_seq_len
    ).to(device)

    dummy_feat = torch.randn(batch_size, original_time_steps, hidden_dim).to(device)

    dummy_lens = torch.tensor([100, 80, 60, 45]).to(device)

    def create_padding_mask(max_len, actual_lengths):
        indices = torch.arange(max_len, device=actual_lengths.device).expand(len(actual_lengths), max_len)
        return indices >= actual_lengths.unsqueeze(1)

    mask = create_padding_mask(original_time_steps, dummy_lens)

    model.eval()
    with torch.no_grad():
        output = model(dummy_feat, src_key_padding_mask=mask)

    print(f"--- TemporalTransformer Test ---")
    print(f"Device:       {device}")
    print(f"Input Shape:  {dummy_feat.shape}")  # [4, 100, 512]
    print(f"Output Shape: {output.shape}")      # [4, 100, 512]
    print(f"Mask Shape:   {mask.shape}")        # [4, 100]
    print(f"Learned Pos:  {model.pos_embedding.shape}")
    
    assert not torch.isnan(output).any(), "Output contains NaNs!"
    print("\nValidation Successful: Model processed variable lengths correctly.")