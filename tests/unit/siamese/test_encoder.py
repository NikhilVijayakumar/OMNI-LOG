# test_encoder.py
import torch
from src.features.siamese.encoder import LogEncoder

# Mock setup
VOCAB_SIZE = 500
EMBED_DIM = 64
HIDDEN_DIM = 128
encoder = LogEncoder(VOCAB_SIZE, EMBED_DIM, HIDDEN_DIM)

# Create two identical logs (as tensors)
log_a = torch.tensor([[10, 20, 30, 0, 0]]) # "INFO Receiving block [PAD] [PAD]"
mask_a = torch.tensor([[1, 1, 1, 0, 0]])

# Pass through encoder
vec_a = encoder(log_a, mask_a)

print(f"Vector Shape: {vec_a.shape}") # Should be [1, 128]
print(f"Vector Norm: {torch.norm(vec_a).item()}") # Should be 1.0 (due to L2 normalization)