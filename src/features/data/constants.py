"""
Constants and Configuration for OMNI-LOG Data Preprocessing.
Defines the shared semantic space and domain-specific log profiles.
"""

# 1. Universal Regex for Tokenization
# Preserves IPs, File Paths, Hex IDs, and structural punctuation.
LOG_TOKEN_REGEX = r'[a-zA-Z0-9_\-\.]+|[:\(\)\[\]=]|\S+'

# 2. Semantic BIO Tags
# Defines the unified entity extraction space across all domains.
TAG_MAP = {
    "<PAD>": 0,
    "O": 1,
    "B-TIME": 2,
    "I-TIME": 3,
    "B-LEVEL": 4,
    "I-LEVEL": 5,
    "B-COMPONENT": 6,
    "I-COMPONENT": 7,
    "B-PARAM": 8,
    "I-PARAM": 9
}

# Inverse map for decoding model output back to text labels
IDX_TO_TAG = {v: k for k, v in TAG_MAP.items()}

# 3. Special Vocabulary Tokens
SPECIAL_TOKENS = {
    "PAD": "<PAD>",
    "UNK": "<UNK>",
    "SOS": "<SOS>",
    "EOS": "<EOS>"
}

# 4. Log Profiles (Domain-Specific Regex)
# Used for log splitting, domain verification, and initial cleaning.
LOG_PROFILES = {
    "syslog": {
        "domains": ["Linux", "OpenSSH"],
        "description": "Standard Syslog format (e.g., Jun 14 15:16:01)",
        "start_regex": r'^[A-Za-z]{3}\s+\d{1,2}\s+\d{2}:\d{2}:\d{2}'
    },
    "java_bigdata": {
        "domains": ["Hadoop", "Zookeeper"],
        "description": "Java Logging Frameworks (e.g., 2015-10-18 18:01:47,978)",
        "start_regex": r'^\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2},\d{3}'
    },
    "apache": {
        "domains": ["Apache"],
        "description": "Web Server Logs (e.g., [Sun Dec 04 04:47:44 2005])",
        "start_regex": r'^\[[A-Za-z]{3}\s+[A-Za-z]{3}\s+\d{2}'
    },
    "proxifier": {
        "domains": ["Proxifier"],
        "description": "Network Proxy Logs (e.g., [10.30 16:49:06])",
        "start_regex": r'^\[\d{1,2}\.\d{1,2}\s+\d{2}:\d{2}:\d{2}\]'
    },
    "android": {
        "domains": ["Android"],
        "description": "Android Logcat (e.g., 03-17 16:13:38.811)",
        "start_regex": r'^\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}\.\d{3}'
    },
    "healthapp": {
        "domains": ["HealthApp"],
        "description": "Mobile Health Application (e.g., 20171223-22:15:29:606|)",
        "start_regex": r'^\d{8}-\d{2}:\d{2}:\d{2}:\d{3}\|'
    },
    "hpc": {
        "domains": ["HPC"],
        "description": "High-Performance Computing State (e.g., 134681 node-246)",
        "start_regex": r'^\d+\s+node-\d+'
    }
}

# 5. Global Domain Mapping
# Flattens all profile domains into a single list for the Chunker's Domain ID.
DOMAINS = sorted([d for profile in LOG_PROFILES.values() for d in profile["domains"]])
DOMAIN_TO_IDX = {name: i for i, name in enumerate(DOMAINS)}

# 6. Preprocessing Defaults
DEFAULT_MAX_SEQ_LEN = 64
DEFAULT_MIN_VOCAB_FREQ = 1