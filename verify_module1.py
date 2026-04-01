import torch
import os
import re
from src.features.data.loader import get_dataloader
from src.features.data.processor import LogProcessor

def verify_tier1():
    print("=== Tier 1: Semantic Tagging Accuracy ===")
    proc = LogProcessor(max_seq_len=64)
    
    # Hadoop Test Case (Java Profile)
    hadoop_log = "2015-10-18 18:01:47,978 INFO [main] org.apache.hadoop.mapreduce.v2.app.MRAppMaster: Created MRAppMaster"
    hadoop_template = "2015-10-18 18:01:47,978 INFO [main] org.apache.hadoop.mapreduce.v2.app.MRAppMaster: <*>"
    
    h_tokens = proc.tokenize(hadoop_log)
    h_temp_tokens = proc.tokenize(hadoop_template)
    h_tags = proc.generate_bio_tags(h_tokens, h_temp_tokens)
    
    print("\nHadoop Token -> Tag Mapping:")
    for t, tag in zip(h_tokens, h_tags):
        print(f"{t:20} -> {tag}")
        
    # Linux Test Case (Syslog Profile)
    linux_log = "Jun 14 15:16:01 java-app-1 kernel: [17] Device failed"
    linux_template = "Jun 14 15:16:01 java-app-1 kernel: [17] <*>"
    
    l_tokens = proc.tokenize(linux_log)
    l_temp_tokens = proc.tokenize(linux_template)
    l_tags = proc.generate_bio_tags(l_tokens, l_temp_tokens)
    
    print("\nLinux Token -> Tag Mapping:")
    for t, tag in zip(l_tokens, l_tags):
        print(f"{t:20} -> {tag}")

    # Check for B-TIME and B-PARAM
    has_time = any("TIME" in t for t in h_tags) and any("TIME" in t for t in l_tags)
    has_param = any("PARAM" in t for t in h_tags) and any("PARAM" in t for t in l_tags)
    
    if has_time: print("\n✅ B-TIME identified in both domains.")
    else: print("\n❌ B-TIME missing in one or more domains.")
    
    if has_param: print("✅ B-PARAM/I-PARAM identified for <*> variables.")
    else: print("❌ B-PARAM/I-PARAM missing for <*> variables.")


def verify_tier2():
    print("\n=== Tier 2: Vocabulary & Padding Integrity ===")
    # Load data to build real vocab
    loader, proc = get_dataloader("data/logs", batch_size=32)
    
    print(f"Vocab Size: {len(proc.vocab)}")
    
    # Pick a short log and a long log
    short_log = "INFO Hello"
    long_log = "DEBUG " + "token " * 50
    
    s_tokens = proc.tokenize(short_log)
    l_tokens = proc.tokenize(long_log)
    
    s_token_ids, s_tag_ids, s_len = proc.numericalize(s_tokens, ["O"]*len(s_tokens))
    l_token_ids, l_tag_ids, l_len = proc.numericalize(l_tokens, ["O"]*len(l_tokens))
    
    print(f"Short Log Length: {s_len}, Tensor Shape: {s_token_ids.shape}")
    print(f"Long Log Length: {l_len}, Tensor Shape: {l_token_ids.shape}")
    
    # Check padding
    assert s_token_ids.shape[0] == 64
    assert l_token_ids.shape[0] == 64
    assert s_token_ids[-1] == 0  # <PAD>
    
    # Check Unknowns
    unk_log = "X-Factor-Log"
    u_tokens = proc.tokenize(unk_log)
    u_token_ids, _, _ = proc.numericalize(u_tokens, ["O"])
    assert u_token_ids[0] == 1  # <UNK>
    
    print("✅ Tier 2 passed: Padding, Length, and UNK mapping are correct.")

def verify_tier3():
    print("\n=== Tier 3: DataLoader Stress Test ===")
    loader, proc = get_dataloader("data/logs", batch_size=32)
    batch = next(iter(loader))
    
    print(f"Batch keys: {batch.keys()}")
    print(f"Tokens Shape: {batch['tokens'].shape}")
    print(f"Tags Shape:   {batch['tags'].shape}")
    
    domain_ids = batch['domain'].unique().tolist()
    print(f"Domain IDs in batch: {domain_ids}")
    
    assert batch['tokens'].shape == (32, 64)
    assert batch['tags'].shape == (32, 64)
    if len(domain_ids) > 1:
        print("✅ Multiple domains found in single batch. Shuffle is working.")
    else:
        print("⚠️ Only one domain found. (Might happen with small batch or lack of variety in first 32 samples)")

if __name__ == "__main__":
    verify_tier1()
    verify_tier2()
    verify_tier3()
