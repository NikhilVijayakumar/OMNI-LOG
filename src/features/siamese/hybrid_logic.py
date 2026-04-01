import torch
from .resolver import TemplateResolver


class HybridParser:
    def __init__(self, chunker, resolver, conf_threshold=0.90):
        """
        Args:
            chunker: Trained Bi-LSTM-CRF model (Module 2).
            resolver: Trained Siamese TemplateResolver (Module 3).
            conf_threshold: The 'Safety Cutoff'. Below this, we fallback to Siamese.
        """
        self.chunker = chunker
        self.resolver = resolver
        self.conf_threshold = conf_threshold

    def parse_log(self, log_line):
        """
        The Master Logic for a single log line.
        """
        # 1. Ask the Chunker first (Fast & Token-Perfect)
        # We assume the chunker has a 'predict' method that returns tags + confidence
        chunker_output, confidence = self.chunker.predict(log_line)

        if confidence >= self.conf_threshold:
            return {
                "method": "Bi-LSTM-CRF",
                "confidence": confidence,
                "structured_data": chunker_output,
                "status": "SUCCESS"
            }

        # 2. Fallback to Siamese Resolver (Semantic & Robust)
        print(f"⚠️ Low confidence ({confidence:.2f}). Triggering Siamese Resolver...")
        siamese_result = self.resolver.resolve(log_line)

        if siamese_result["match_found"]:
            return {
                "method": "Siamese-Resolver",
                "confidence": siamese_result["similarity"],
                "template_id": siamese_result["template_id"],
                "template_text": siamese_result["template_text"],
                "status": "RESOLVED"
            }

        # 3. Final Fallback (Unknown Pattern)
        return {
            "method": "None",
            "status": "UNKNOWN_PATTERN",
            "raw_log": log_line
        }