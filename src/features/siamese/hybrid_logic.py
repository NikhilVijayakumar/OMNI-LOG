import torch
from .resolver import TemplateResolver
from features.data.rule_parser import parse as rule_parse


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


def _rule_structured(log_line: str) -> dict:
    """Run the rule-based parser and return a structured dict in the standard schema."""
    r = rule_parse(log_line)
    params_str = r.get("params") or ""
    return {
        "time":      r.get("time"),
        "level":     r.get("level"),
        "component": r.get("component"),
        "params":    params_str.split() if params_str else [],
    }


def _fill_gaps(structured: dict, rule: dict) -> None:
    """Supplement model output with rule-based values wherever the model is incomplete."""
    # For time: rule parser extracts the full domain-format timestamp; use it when it is more complete.
    rule_time = rule.get("time")
    if rule_time and len(rule_time) > len(structured.get("time") or ""):
        structured["time"] = rule_time
    if structured.get("level") is None and rule.get("level"):
        structured["level"] = rule["level"]
    if structured.get("component") is None and rule.get("component"):
        structured["component"] = rule["component"]
    if not structured.get("params") and rule.get("params"):
        structured["params"] = rule["params"]


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
        Rule-based parsing runs for every line and fills any gaps left by the models.
        """
        # Always run rule-based parser — zero cost, ensures no field is ever null if avoidable
        rule = _rule_structured(log_line)

        # 0. Preprocessing for the Neural Chunker
        tokens = self.processor.tokenize(log_line)
        if len(tokens) == 0:
            return {"raw": log_line, "structured": rule,
                    "metadata": {"method": "Rule-Based", "status": "RULE_PARSED"}}

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
            # Fill any field the model left null using rule-based output
            _fill_gaps(out["structured"], rule)
            out["metadata"] = {"method": "Bi-LSTM-CRF", "confidence": round(conf_val, 6), "status": "SUCCESS"}
            return out

        # 2. Fallback to Siamese Resolver (Semantic & Robust)
        siamese_result = self.resolver.resolve(log_line)

        if siamese_result["match_found"]:
            # Rule-based gives the 4 fields; add template metadata on top
            siamese_structured = dict(rule)
            siamese_structured["template_id"] = siamese_result["template_id"]
            siamese_structured["template_text"] = siamese_result["template_text"]
            return {
                "raw": log_line,
                "structured": siamese_structured,
                "metadata": {"method": "Siamese-Resolver",
                             "confidence": round(siamese_result["similarity"], 6),
                             "status": "RESOLVED"},
            }

        # 3. Final Fallback: rule-based only — never return null structured when avoidable
        has_data = any(rule.get(k) is not None for k in ("time", "level", "component")) or bool(rule.get("params"))
        status = "RULE_PARSED" if has_data else "UNKNOWN_PATTERN"
        return {
            "raw": log_line,
            "structured": rule if has_data else None,
            "metadata": {"method": "Rule-Based", "status": status},
        }