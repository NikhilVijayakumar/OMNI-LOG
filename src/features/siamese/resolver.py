import torch
import torch.nn.functional as F
from src.features.data.processor import LogProcessor


class TemplateResolver:
    def __init__(self, encoder, processor, device="cpu"):
        """
        Args:
            encoder: The trained LogEncoder from Phase 1.
            processor: The LogProcessor from Module 1 (for tokenization).
        """
        self.encoder = encoder.to(device)
        self.processor = processor
        self.device = device

        # Internal storage for the "Knowledge Base"
        self.template_vectors = None  # Tensor Matrix [Num_Templates, Hidden_Dim]
        self.template_metadata = []  # List of strings/IDs
        self.threshold = 0.85  # Minimum Cosine Similarity to accept a match

    def build_library(self, templates_dict):
        """
        Encodes all known templates into the vector space.
        templates_dict: {template_id: "Receiving block <*> from <*>", ...}
        """
        self.encoder.eval()
        ids = []
        texts = []
        vectors = []

        with torch.no_grad():
            for t_id, text in templates_dict.items():
                # 1. Tokenize and Numericalize the template itself
                tokens = self.processor.tokenize(text)
                token_ids, _, length = self.processor.numericalize(tokens, ["O"] * len(tokens))

                # 2. Create Mask
                mask = (token_ids != 0).unsqueeze(0).to(self.device)
                token_ids = token_ids.unsqueeze(0).to(self.device)

                # 3. Encode to Vector
                vec = self.encoder(token_ids, mask)

                vectors.append(vec)
                ids.append(t_id)
                texts.append(text)

        self.template_vectors = torch.cat(vectors, dim=0)  # [N, Hidden_Dim]
        self.template_metadata = {"ids": ids, "texts": texts}
        print(f"✅ Template Library built with {len(ids)} vectors.")

    def resolve(self, log_line):
        """
        Finds the closest template for a raw log line.
        """
        self.encoder.eval()
        with torch.no_grad():
            # 1. Preprocess input log
            tokens = self.processor.tokenize(log_line)
            token_ids, _, _ = self.processor.numericalize(tokens, ["O"] * len(tokens))
            mask = (token_ids != 0).unsqueeze(0).to(self.device)
            token_ids = token_ids.unsqueeze(0).to(self.device)

            # 2. Encode input log
            log_vector = self.encoder(token_ids, mask)  # [1, Hidden_Dim]

            # 3. Compute Cosine Similarity against ALL templates at once
            # Since vectors are L2-normalized, similarity is just a matrix multiplication
            similarities = torch.matmul(log_vector, self.template_vectors.T).squeeze(0)

            # 4. Find the best match
            max_sim, best_idx = torch.max(similarities, dim=0)

            result = {
                "template_id": self.template_metadata["ids"][best_idx.item()],
                "template_text": self.template_metadata["texts"][best_idx.item()],
                "similarity": max_sim.item(),
                "match_found": max_sim.item() >= self.threshold
            }

            return result