from dataclasses import dataclass
import yaml

@dataclass
class BatchConfig:
    batch_size: int = 64
    write_batch_size: int = 500
    max_seq_len: int = 64
    confidence_threshold: float = 0.90
    similarity_threshold: float = 0.85
    
    @classmethod
    def from_yaml(cls, yaml_path: str):
        import os
        if not os.path.exists(yaml_path):
            return cls()
            
        with open(yaml_path, 'r') as f:
            cfg = yaml.safe_load(f)
            
        # Try to map fields from yaml if possible
        b_size = cfg.get('train', {}).get('batch_size', 64)
        max_seq = cfg.get('preprocessing', {}).get('max_seq_len', 64)
        
        return cls(
            batch_size=b_size,
            max_seq_len=max_seq,
        )
