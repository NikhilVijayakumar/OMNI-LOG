import torch
from features.siamese.encoder import LogEncoder

def test_encoder_l2_normalization():
    # Mock setup
    VOCAB_SIZE = 500
    EMBED_DIM = 64
    HIDDEN_DIM = 128
    
    encoder = LogEncoder(vocab_size=VOCAB_SIZE, embedding_dim=EMBED_DIM, hidden_dim=HIDDEN_DIM)

    # Create two identical logs (as tensors)
    log_a = torch.tensor([[10, 20, 30, 0, 0]]) # "INFO Receiving block [PAD] [PAD]"
    mask_a = torch.tensor([[1, 1, 1, 0, 0]])

    # Pass through encoder
    vec_a = encoder(log_a, mask_a)
    
    print(f"Vector Shape: {vec_a.shape}") # Should be [1, 128]
    assert vec_a.shape == (1, 128)
    
    norm = torch.norm(vec_a).item()
    print(f"Vector Norm: {norm}") # Should be 1.0 (due to L2 normalization)
    assert abs(norm - 1.0) < 1e-4
    
    print("[OK] Encoder L2 normalization test passed!")

if __name__ == "__main__":
    test_encoder_l2_normalization()