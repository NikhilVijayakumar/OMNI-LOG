import torch
from .resolver import TemplateResolver


def _build_structured(log_line, tokens, tags):
    """Converts token-tag pairs into named entity fields."""
    time_tokens, level_tokens, component_tokens, param_tokens = [], [], [], []
    for token, tag in zip(tokens, tags):
        if tag in ("B-TIME", "I-TIME"):
            time_tokens.append(token)
        elif tag in ("B-LEVEL", "I-LEVEL"):
            level_tokens.append(token)
        elif tag in ("B-COMPONENT", "I-COMPONENT"):
            component_tokens.append(token)
        elif tag in ("B-PARAM", "I-PARAM"):
            param_tokens.append(token)
    return {
        "raw": log_line,
        "structured": {
            "time": " ".join(time_tokens) if time_tokens else None,
            "level": " ".join(level_tokens) if level_tokens else None,
            "component": " ".join(component_tokens) if component_tokens else None,
            "params": param_tokens if param_tokens else [],
        },
    }


class HybridParser:
    def __init__(self, chunker, resolver, processor, conf_threshold=0.90):
        """
        Args:
            chunker: Trained Bi-LSTM-CRF model (Module 2).
            resolver: Trained Siamese TemplateResolver (Module 3).
            processor: The LogProcessor mapping logic text tokenizations.
            conf_threshold: The 'Safety Cutoff'. Below this, we fallback to Siamese.
        """
        self.chunker = chunker
        self.resolver = resolver
        self.processor = processor
        self.conf_threshold = conf_threshold
        self.device = next(self.chunker.parameters()).device

    def parse_log(self, log_line):
        """
        The Master Logic for a single log line.
        """
        # 0. Preprocessing for the Neural Chunker
        tokens = self.processor.tokenize(log_line)
        if len(tokens) == 0:
            return {"raw": log_line, "structured": None,
                    "metadata": {"method": "None", "status": "UNKNOWN_PATTERN"}}

        token_ids, _, _ = self.processor.numericalize(tokens, ["O"] * len(tokens))
        mask = (token_ids != 0).unsqueeze(0).to(self.device)
        token_ids = token_ids.unsqueeze(0).to(self.device)

        # 1. Ask the Chunker first (Fast & Token-Perfect)
        with torch.no_grad():
            chunker_output, confidence = self.chunker.predict(token_ids, mask)

        conf_val = confidence[0].item() if hasattr(confidence[0], 'item') else float(confidence[0])

        if conf_val >= self.conf_threshold:
            predicted_tags = [self.processor.idx2tag.get(idx, "O") for idx in chunker_output[0]]
            seq_len = min(len(tokens), len(predicted_tags))
            out = _build_structured(log_line, tokens[:seq_len], predicted_tags[:seq_len])
            out["metadata"] = {"method": "Bi-LSTM-CRF", "confidence": round(conf_val, 6), "status": "SUCCESS"}
            return out

        # 2. Fallback to Siamese Resolver (Semantic & Robust)
        siamese_result = self.resolver.resolve(log_line)

        if siamese_result["match_found"]:
            return {
                "raw": log_line,
                "structured": {"template_id": siamese_result["template_id"],
                               "template_text": siamese_result["template_text"]},
                "metadata": {"method": "Siamese-Resolver",
                             "confidence": round(siamese_result["similarity"], 6),
                             "status": "RESOLVED"},
            }

        # 3. Final Fallback (Unknown Pattern)
        return {
            "raw": log_line,
            "structured": None,
            "metadata": {"method": "None", "status": "UNKNOWN_PATTERN"},
        }