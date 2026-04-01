# test_model.py
import torch
from src.features.chunker.model import BiLSTM_CRF
from src.features.data.constants import TAG_MAP

# Mock dimensions
VOCAB_SIZE = 1000
EMBED_DIM = 128
HIDDEN_DIM = 256

model = BiLSTM_CRF(VOCAB_SIZE, TAG_MAP, EMBED_DIM, HIDDEN_DIM)

# Mock input (Batch of 2 logs, 10 tokens each)
dummy_input = torch.randint(0, VOCAB_SIZE, (2, 10))
dummy_tags = torch.randint(0, len(TAG_MAP), (2, 10))
mask = torch.ones((2, 10), dtype=torch.uint8)

# Test Forward (Loss)
loss = model(dummy_input, dummy_tags, mask=mask)
print(f"Training Loss: {loss.item()}")

# Test Decode (Inference)
prediction = model.decode(dummy_input, mask=mask)
print(f"Predicted Tags for Log 1: {prediction[0]}")