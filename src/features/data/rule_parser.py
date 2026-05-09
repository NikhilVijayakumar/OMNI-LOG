# src/features/data/rule_parser.py
"""
Rule-based log parser using domain-specific regex patterns.
Guarantees structured extraction for all 8 LogHub formats.
Used as a reliable fallback/supplement when the neural model fails.
"""
import re

_LEVEL_WORDS = {"INFO", "DEBUG", "WARN", "WARNING", "ERROR", "FATAL", "TRACE", "NOTICE", "SEVERE"}

# (format_name, compiled_regex)
# Groups: time, level (optional), component (optional), params
_PATTERNS = [
    # Hadoop: `2015-10-18 18:01:47,978 INFO [main] org.apache.Class: message`
    ("hadoop", re.compile(
        r'^(?P<time>\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2},\d{3})'
        r'\s+(?P<level>INFO|DEBUG|WARN|WARNING|ERROR|FATAL|TRACE)'
        r'\s+\[[^\]]*\]'
        r'\s+(?P<component>\S+?):'
        r'\s*(?P<params>.*)$',
        re.DOTALL
    )),
    # Zookeeper: `2015-07-29 17:41:44,747 - INFO  [thread:Class@line] - message`
    # Thread string may contain nested brackets, so match greedily up to the final ] - separator.
    ("zookeeper", re.compile(
        r'^(?P<time>\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2},\d{3})'
        r'\s+-\s+(?P<level>INFO|DEBUG|WARN|WARNING|ERROR|FATAL|TRACE)\s+'
        r'\[(?P<component>.+)\]'
        r'\s+-\s+(?P<params>.*)$',
        re.DOTALL
    )),
    # Android: `03-17 16:13:38.811  1702  2395 D WindowManager: message`
    ("android", re.compile(
        r'^(?P<time>\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}\.\d+)'
        r'\s+\d+\s+\d+'
        r'\s+(?P<level>[VDIWEF])'
        r'\s+(?P<component>[^:]+?):'
        r'\s*(?P<params>.*)$',
        re.DOTALL
    )),
    # Linux/OpenSSH Syslog: `Jun 14 15:16:01 combo sshd(pam_unix)[19939]: message`
    ("syslog", re.compile(
        r'^(?P<time>[A-Za-z]{3}\s+\d{1,2}\s+\d{2}:\d{2}:\d{2})'
        r'\s+\S+'
        r'\s+(?P<component>\S+?)(?:\(\S+?\))?(?:\[\d+\])?:'
        r'\s*(?P<params>.*)$',
        re.DOTALL
    )),
    # Apache: `[Sun Dec 04 04:47:44 2005] [notice] message`
    ("apache", re.compile(
        r'^\[(?P<time>[A-Za-z]{3}\s+[A-Za-z]{3}\s+\d{2}\s+\d{2}:\d{2}:\d{2}\s+\d{4})\]'
        r'\s+\[(?P<level>\w+)\]'
        r'\s+(?P<params>.*)$',
        re.DOTALL
    )),
    # HealthApp: `20171223-22:15:29:606|Step_LSC|30002312|message`
    ("healthapp", re.compile(
        r'^(?P<time>\d{8}-\d{2}:\d{2}:\d{2}:\d{3})'
        r'\|(?P<component>[^|]+)'
        r'\|[^|]+'
        r'\|(?P<params>.*)$',
        re.DOTALL
    )),
    # Proxifier: `[10.30 16:49:06] chrome.exe - message`
    ("proxifier", re.compile(
        r'^\[(?P<time>[\d\.]+\s+\d{2}:\d{2}:\d{2})\]'
        r'\s+(?P<component>[^\s-]+)'
        r'\s+-\s+(?P<params>.*)$',
        re.DOTALL
    )),
    # HPC: `134681 node-246 unix.hw state_change.unavailable 1077804742 1 message`
    # Fields: jobid  node  event_class  event_type  unix_timestamp  count  message
    ("hpc", re.compile(
        r'^\d+'
        r'\s+(?P<component>node-\d+)'
        r'\s+\S+'          # event_class  (e.g. unix.hw)
        r'\s+\S+'          # event_type   (e.g. state_change.unavailable)
        r'\s+(?P<time>\d+)'
        r'\s+\d+'
        r'\s+(?P<params>.*)$',
        re.DOTALL
    )),
]


def parse(log_line: str) -> dict:
    """
    Try each domain pattern in order. Returns a dict with:
      time, level, component, params
    Any field may be None if not found.
    """
    for _name, pattern in _PATTERNS:
        m = pattern.match(log_line.rstrip())
        if m:
            gd = m.groupdict()
            return {
                "time":      gd.get("time"),
                "level":     gd.get("level"),
                "component": gd.get("component"),
                "params":    gd.get("params", ""),
            }
    return _fallback_extract(log_line)


def _fallback_extract(log_line: str) -> dict:
    """
    Last-resort token scan: find the level word and treat everything
    before it as time+component context, everything after as params.
    """
    tokens = log_line.split()
    time_val = level_val = component_val = None
    param_start = 0

    for i, tok in enumerate(tokens):
        if tok.upper() in _LEVEL_WORDS:
            level_val = tok
            # Try to get component from the token right before level
            if i > 0 and tokens[i - 1] not in {"[", "]", ":", "-", "|"}:
                component_val = tokens[i - 1].strip("[]:")
            param_start = i + 1
            break
        # ISO date
        if re.match(r'^\d{4}-\d{2}-\d{2}$', tok) and time_val is None:
            time_val = tok
            if i + 1 < len(tokens) and re.match(r'^\d{2}:\d{2}:\d{2}', tokens[i + 1]):
                time_val += " " + tokens[i + 1]

    params = " ".join(tokens[param_start:]) if param_start else log_line
    return {
        "time":      time_val,
        "level":     level_val,
        "component": component_val,
        "params":    params,
    }
