import torch
import torch.nn as nn
import torch.nn.functional as F


class LogEncoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, dropout=0.1):
        """
        OMNI-LOG Siamese Encoder
        Transforms a log sequence into a fixed-size semantic vector.
        """
        super(LogEncoder, self).__init__()

        # 1. Shared Embedding Layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # 2. Bi-LSTM for Contextual Encoding
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim // 2,
            bidirectional=True,
            batch_first=True,
            dropout=dropout
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        """
        Args:
            x: [batch_size, seq_len] - Token IDs
            mask: [batch_size, seq_len] - 1 for real tokens, 0 for padding
        Returns:
            Vector: [batch_size, hidden_dim] - Normalized Log Embedding
        """
        # 1. Embed and LSTM
        embeds = self.embedding(x)  # [B, L, E]
        lstm_out, _ = self.lstm(embeds)  # [B, L, H]
        lstm_out = self.dropout(lstm_out)

        # 2. Masked Mean Pooling (Crucial for Siamese Networks)
        # We average only the real tokens, ignoring the zeros (padding)
        mask_expanded = mask.unsqueeze(-1).expand(lstm_out.size()).float()

        # Sum all hidden states that are NOT padding
        sum_embeddings = torch.sum(lstm_out * mask_expanded, dim=1)

        # Divide by the actual length of each log
        actual_lengths = torch.clamp(mask_expanded.sum(1), min=1e-9)
        pooled_output = sum_embeddings / actual_lengths

        # 3. L2 Normalization
        # This makes Cosine Similarity equal to a simple Dot Product
        return F.normalize(pooled_output, p=2, dim=1)


class SiameseNet(nn.Module):
    def __init__(self, encoder):
        super(SiameseNet, self).__init__()
        self.encoder = encoder

    def forward(self, log_a, mask_a, log_b, mask_b):
        """
        Computes the similarity between two logs.
        Used primarily during training or manual comparison.
        """
        vec_a = self.encoder(log_a, mask_a)
        vec_b = self.encoder(log_b, mask_b)

        # Cosine similarity is the dot product of normalized vectors
        return torch.sum(vec_a * vec_b, dim=1)