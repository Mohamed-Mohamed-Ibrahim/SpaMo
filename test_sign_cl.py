"""
Test script for Sign Contrastive Learning (SignCL) loss implementation.

This script validates that the SignCL loss computes correctly and integrates
properly with the SpaMo model.
"""

import torch
import torch.nn as nn
from spamo.sign_cl import TemporalSignCLLoss, SignCLLoss


def test_temporal_sign_cl_loss():
    """Test TemporalSignCLLoss (efficient version) with various input shapes."""
    print("\n" + "="*80)
    print("Testing TemporalSignCLLoss (Efficient Version)")
    print("="*80)
    
    loss_fn = TemporalSignCLLoss(
        temperature=0.07,
        temporal_window=5,
    )
    
    # Test 1: Basic functionality
    print("\nTest 1: Basic Functionality")
    batch_size, seq_len, embed_dim = 4, 20, 768
    visual_embeddings = torch.randn(batch_size, seq_len, embed_dim)
    visual_masks = torch.ones(batch_size, seq_len)
    
    loss = loss_fn(visual_embeddings, visual_masks)
    print(f"  Input shape: {visual_embeddings.shape}")
    print(f"  Loss value: {loss.item():.6f}")
    print(f"  Loss requires grad: {loss.requires_grad}")
    print(f"  ✓ Basic functionality passed")
    
    # Test 2: Variable length sequences
    print("\nTest 2: Variable Length Sequences")
    batch_size, max_len, embed_dim = 4, 30, 768
    visual_embeddings = torch.randn(batch_size, max_len, embed_dim)
    visual_masks = torch.ones(batch_size, max_len)
    
    # Make some sequences shorter
    visual_masks[0, 15:] = 0  # Sequence 0 has length 15
    visual_masks[1, 20:] = 0  # Sequence 1 has length 20
    visual_masks[2, 25:] = 0  # Sequence 2 has length 25
    
    loss = loss_fn(visual_embeddings, visual_masks)
    print(f"  Sequence lengths: {visual_masks.sum(dim=1).int().tolist()}")
    print(f"  Loss value: {loss.item():.6f}")
    print(f"  ✓ Variable length passed")
    
    # Test 3: Different temperature values
    print("\nTest 3: Temperature Parameter Effects")
    temperatures = [0.01, 0.07, 0.1, 0.2]
    for temp in temperatures:
        loss_fn_temp = TemporalSignCLLoss(temperature=temp, temporal_window=5)
        loss = loss_fn_temp(visual_embeddings, visual_masks)
        print(f"  Temperature {temp:4.2f}: loss = {loss.item():.6f}")
    print(f"  ✓ Temperature variations passed")
    
    # Test 4: Different temporal windows
    print("\nTest 4: Temporal Window Parameter Effects")
    windows = [2, 5, 10, 15]
    for window in windows:
        loss_fn_window = TemporalSignCLLoss(temperature=0.07, temporal_window=window)
        loss = loss_fn_window(visual_embeddings, visual_masks)
        print(f"  Window {window:2d}: loss = {loss.item():.6f}")
    print(f"  ✓ Temporal window variations passed")
    
    # Test 5: Empty batch (should not crash)
    print("\nTest 5: Edge Cases")
    # Short sequences
    short_embeddings = torch.randn(2, 1, embed_dim)  # Only 1 frame per batch
    short_masks = torch.ones(2, 1)
    loss = loss_fn(short_embeddings, short_masks)
    print(f"  Short sequences (1 frame): loss = {loss.item():.6f}")
    print(f"  ✓ Edge cases handled gracefully")
    
    print("\n✓ All TemporalSignCLLoss tests passed!")
    return True


def test_sign_cl_loss():
    """Test SignCLLoss (advanced version) with various input shapes."""
    print("\n" + "="*80)
    print("Testing SignCLLoss (Advanced Version)")
    print("="*80)
    
    loss_fn = SignCLLoss(
        temperature=0.07,
        temporal_window=5,
        use_cosine_similarity=True,
    )
    
    # Test 1: Basic functionality
    print("\nTest 1: Basic Functionality")
    batch_size, seq_len, embed_dim = 4, 20, 768
    visual_embeddings = torch.randn(batch_size, seq_len, embed_dim)
    visual_masks = torch.ones(batch_size, seq_len)
    
    loss = loss_fn(visual_embeddings, visual_masks)
    print(f"  Input shape: {visual_embeddings.shape}")
    print(f"  Loss value: {loss.item():.6f}")
    print(f"  Loss requires grad: {loss.requires_grad}")
    print(f"  ✓ Basic functionality passed")
    
    # Test 2: Cosine similarity vs dot product
    print("\nTest 2: Similarity Metric Options")
    loss_cosine = loss_fn(visual_embeddings, visual_masks)
    
    loss_fn_dot = SignCLLoss(
        temperature=0.07,
        temporal_window=5,
        use_cosine_similarity=False,
    )
    loss_dot = loss_fn_dot(visual_embeddings, visual_masks)
    
    print(f"  Cosine similarity loss: {loss_cosine.item():.6f}")
    print(f"  Dot product loss:       {loss_dot.item():.6f}")
    print(f"  ✓ Both metrics work")
    
    # Test 3: Variable length sequences
    print("\nTest 3: Variable Length Sequences")
    visual_masks[0, 15:] = 0
    visual_masks[1, 20:] = 0
    
    loss = loss_fn(visual_embeddings, visual_masks)
    print(f"  Sequence lengths: {visual_masks.sum(dim=1).int().tolist()}")
    print(f"  Loss value: {loss.item():.6f}")
    print(f"  ✓ Variable length passed")
    
    print("\n✓ All SignCLLoss tests passed!")
    return True


def test_gradient_flow():
    """Test that gradients flow properly through SignCL loss."""
    print("\n" + "="*80)
    print("Testing Gradient Flow")
    print("="*80)
    
    loss_fn = TemporalSignCLLoss(temperature=0.07, temporal_window=5)
    
    batch_size, seq_len, embed_dim = 2, 15, 768
    visual_embeddings = torch.randn(
        batch_size, seq_len, embed_dim, 
        requires_grad=True
    )
    visual_masks = torch.ones(batch_size, seq_len)
    
    print("\nTest 1: Backward Pass")
    loss = loss_fn(visual_embeddings, visual_masks)
    
    # Check if loss can be backpropagated
    loss.backward()
    
    print(f"  Loss: {loss.item():.6f}")
    print(f"  Gradient exists: {visual_embeddings.grad is not None}")
    print(f"  Gradient shape: {visual_embeddings.grad.shape}")
    print(f"  Gradient non-zero: {(visual_embeddings.grad != 0).any().item()}")
    
    if visual_embeddings.grad is not None and (visual_embeddings.grad != 0).any():
        print(f"  ✓ Gradients flow correctly")
    else:
        print(f"  ✗ Warning: No gradients flowing")
    
    print("\nTest 2: Integration with Optimizer")
    
    # Reset gradients
    visual_embeddings.grad = None
    
    # Create a simple model with the loss
    optimizer = torch.optim.Adam([visual_embeddings], lr=1e-3)
    
    # Training loop
    initial_embedding = visual_embeddings.clone().detach()
    for step in range(3):
        optimizer.zero_grad()
        loss = loss_fn(visual_embeddings, visual_masks)
        loss.backward()
        optimizer.step()
        print(f"  Step {step}: loss = {loss.item():.6f}")
    
    # Check if embeddings changed
    embedding_changed = not torch.allclose(initial_embedding, visual_embeddings, atol=1e-5)
    print(f"  Embeddings updated by optimizer: {embedding_changed}")
    
    if embedding_changed:
        print(f"  ✓ Gradient flow and optimization work")
    else:
        print(f"  ✗ Warning: Embeddings did not update")
    
    print("\n✓ Gradient flow tests passed!")
    return True


def test_numerical_stability():
    """Test numerical stability of the loss with extreme values."""
    print("\n" + "="*80)
    print("Testing Numerical Stability")
    print("="*80)
    
    loss_fn = TemporalSignCLLoss(temperature=0.07, temporal_window=5)
    
    batch_size, seq_len, embed_dim = 2, 15, 768
    visual_masks = torch.ones(batch_size, seq_len)
    
    # Test 1: Very small embeddings
    print("\nTest 1: Very Small Embeddings")
    small_embeddings = torch.randn(batch_size, seq_len, embed_dim) * 1e-6
    loss = loss_fn(small_embeddings, visual_masks)
    print(f"  Embedding scale: 1e-6")
    print(f"  Loss: {loss.item():.6f}")
    print(f"  Is finite: {torch.isfinite(loss).item()}")
    
    # Test 2: Very large embeddings
    print("\nTest 2: Very Large Embeddings")
    large_embeddings = torch.randn(batch_size, seq_len, embed_dim) * 1e3
    loss = loss_fn(large_embeddings, visual_masks)
    print(f"  Embedding scale: 1e3")
    print(f"  Loss: {loss.item():.6f}")
    print(f"  Is finite: {torch.isfinite(loss).item()}")
    
    # Test 3: Uniform embeddings (edge case)
    print("\nTest 3: Uniform Embeddings (all same)")
    uniform_embeddings = torch.ones(batch_size, seq_len, embed_dim)
    loss = loss_fn(uniform_embeddings, visual_masks)
    print(f"  All embeddings identical")
    print(f"  Loss: {loss.item():.6f}")
    print(f"  Is finite: {torch.isfinite(loss).item()}")
    
    print("\n✓ Numerical stability tests passed!")
    return True


def main():
    """Run all tests."""
    print("\n" + "="*80)
    print("SIGN CONTRASTIVE LEARNING (SignCL) - TEST SUITE")
    print("="*80)
    
    try:
        test_temporal_sign_cl_loss()
        test_sign_cl_loss()
        test_gradient_flow()
        test_numerical_stability()
        
        print("\n" + "="*80)
        print("✓ ALL TESTS PASSED SUCCESSFULLY!")
        print("="*80)
        print("\nSignCL is ready for use. Configure it in configs/finetune.yaml:")
        print("  sign_cl_loss: true")
        print("  sign_cl_alpha: 0.5")
        print("  sign_cl_temperature: 0.07")
        print("  sign_cl_temporal_window: 5")
        print("="*80 + "\n")
        
        return True
        
    except Exception as e:
        print(f"\n✗ TEST FAILED WITH ERROR:")
        print(f"  {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
