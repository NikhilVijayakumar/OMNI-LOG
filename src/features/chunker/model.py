import torch
import torch.nn as nn
from TorchCRF import CRF



class BiLSTM_CRF(nn.Module):
    def __init__(self, vocab_size, tag_to_ix, embedding_dim, hidden_dim, num_layers=1, dropout=0.1):
        """
        OMNI-LOG Chunker Architecture

        Args:
            vocab_size: Number of unique tokens in the universal vocabulary.
            tag_to_ix: Dictionary mapping BIO tags to indices (from constants.py).
            embedding_dim: Dimension of token embeddings.
            hidden_dim: Dimension of LSTM hidden states.
            num_layers: Number of LSTM layers.
            dropout: Regularization probability.
        """
        super(BiLSTM_CRF, self).__init__()
        self.tagset_size = len(tag_to_ix)

        # 1. Embedding Layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # 2. Bi-LSTM Layer
        # batch_first=True allows input shape (batch, seq, feature)
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim // 2,  # Divide by 2 because bidirectional=True doubles it
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # 3. Linear Dropout for regularization
        self.dropout = nn.Dropout(dropout)

        # 4. Hidden to Tag Projection
        # Maps LSTM output to the number of possible BIO tags
        self.hidden2tag = nn.Linear(hidden_dim, self.tagset_size)

        # 5. CRF Layer
        # Handles global sequence optimization and transition constraints
        self.crf = CRF(self.tagset_size)

    def _get_lstm_features(self, sentence):
        """Passes input through Embedding and Bi-LSTM layers."""
        embeds = self.embedding(sentence)
        lstm_out, _ = self.lstm(embeds)
        lstm_out = self.dropout(lstm_out)
        tag_space = self.hidden2tag(lstm_out)
        return tag_space

    def forward(self, sentence, tags, mask=None):
        """
        Used during TRAINING.
        Returns the negative log-likelihood loss.
        """
        features = self._get_lstm_features(sentence)
        # CRF returns the log-likelihood as a vector; we negate it and mean it for the loss
        log_likelihood = self.crf(features, tags, mask=mask)
        return -log_likelihood.mean()

    def decode(self, sentence, mask=None):
        """
        Used during INFERENCE / VALIDATION.
        Returns the most likely tag sequence using the Viterbi algorithm.
        """
        features = self._get_lstm_features(sentence)
        # Returns a list of lists (best path for each sequence in batch)
        return self.crf.viterbi_decode(features, mask=mask)


    def get_confidence(self, sentence, mask=None):
        """
        Calculates the normalized path score.
        High score = high confidence in the sequence structure.
        """
        features = self._get_lstm_features(sentence)
        # The raw score of the best path
        best_path_scores = self.crf.compute_partitions(features, mask=mask)
        # Normalize by sequence length to get a 'confidence' proxy
        return best_path_scores / sentence.size(1)