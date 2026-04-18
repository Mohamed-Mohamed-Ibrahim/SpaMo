import torch
import torch.nn as nn

class AdaptiveFusion(nn.Module):
    """
    Adaptive Fusion Mechanism inspired by AVRET (Updated for 3 Inputs).
    
    Formula: fused = (I1 + I2 + I3) + (λ1*I1 + λ2*I2 + λ3*I3)
    """
    def __init__(self, input_size_1=512, input_size_2=512, input_size_3=512, output_size=3, bias=False, use_softmax=False, verbose=False):
        """
        Args:
            input_size_1: dimensionality of first input
            input_size_2: dimensionality of second input
            input_size_3: dimensionality of third input
            output_size: number of adaptive weight channels (default 3 for λ₁, λ₂, λ₃)
            bias: whether to use bias in linear layers
            use_softmax: if True, use softmax (competitive/normalized); if False, use sigmoid (independent gating)
            verbose: if True, print tensor shapes at each step
        """
        super(AdaptiveFusion, self).__init__()
        self.verbose = verbose
        self.activation = nn.Softmax(dim=2) if use_softmax else nn.Sigmoid()
        self.weight_input_1 = nn.Linear(input_size_1, output_size, bias=bias)
        self.weight_input_2 = nn.Linear(input_size_2, output_size, bias=bias)
        self.weight_input_3 = nn.Linear(input_size_3, output_size, bias=bias)
        self.layer_norm = nn.LayerNorm(input_size_1, eps=1e-5)
        
    def forward(self, input_1, input_2, input_3):
        """
        Fuse two input representations adaptively.
        
        Args:
            input_1: First input tensor [B, T, D]
            input_2: Second input tensor [B, T, D] (same shape as input_1)
            
        Returns:
            Fused representation [B, T, D]
        """
        weight_sum = (self.weight_input_1(input_1) +
                      self.weight_input_2(input_2) +
                      self.weight_input_3(input_3))

        fm = self.activation(weight_sum)

        lambda1 = fm[:, :, 0].unsqueeze(-1)  # [B, T, 1]
        lambda2 = fm[:, :, 1].unsqueeze(-1)  # [B, T, 1]
        lambda3 = fm[:, :, 2].unsqueeze(-1)  # [B, T, 1]

        fused_output = (input_1 + input_2 + input_3) + \
                       torch.mul(lambda1, input_1) + \
                       torch.mul(lambda2, input_2) + \
                       torch.mul(lambda3, input_3)

        fused_output = self.layer_norm(fused_output)

        if self.verbose:
            print(f"[AdaptiveFusion] input_1:        {input_1.shape}")
            print(f"[AdaptiveFusion] weight_sum:     {weight_sum.shape}")
            print(f"[AdaptiveFusion] fm ({self.activation.__class__.__name__}):   {fm.shape}")
            print(f"[AdaptiveFusion] lambda1:        {lambda1.shape}")
            print(f"[AdaptiveFusion] fused_output:   {fused_output.shape}")

        return fused_output

if __name__ == "__main__":
    batch_size = 2
    seq_len = 10
    dim = 32

    model = AdaptiveFusion(input_size_1=dim, input_size_2=dim, input_size_3=dim, verbose=True)

    i1 = torch.randn(batch_size, seq_len, dim)
    i2 = torch.randn(batch_size, seq_len, dim)
    i3 = torch.randn(batch_size, seq_len, dim)

    output = model(i1, i2, i3)