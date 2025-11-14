#!/usr/bin/env python3
import re
import sys


def normalize(raw_tag: str) -> str:
    raw = raw_tag.lstrip("v")
    raw = raw.split("+", 1)[0]
    base, norm = raw, raw
    if "-" in raw:
        base, suf = raw.split("-", 1)
        suf = suf.lower()
        m = re.match(r"^(alpha|beta|rc|a|b|dev)[\.-]?(\d+)?$", suf)
        if m:
            label, num = m.group(1), m.group(2)
            label = {"alpha": "a", "beta": "b", "rc": "rc", "a": "a", "b": "b", "dev": "dev"}[label]
            if num is None:
                num = "0"
            norm = f"{base}{label}{num}"
    if not re.match(r"^\d", norm):
        norm = raw
    return norm


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("", end="")
        sys.exit(0)
    print(normalize(sys.argv[1]), end="")

