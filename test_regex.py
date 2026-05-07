import re
import sys

def test_chassis(text):
    print("Testing Chassis...")
    _CHASSIS_RE = re.compile(r"([A-Z]{2,10}\d{2,6}[A-Z]?)\s*[-—–_]{1,3}\s*([0-9A-Z\s]{5,15})\b", re.IGNORECASE)
    for m in _CHASSIS_RE.finditer(text):
        print("MATCH:", m.group(1), "---", m.group(2))

def test_expiry(text):
    print("Testing Expiry...")
    m = re.search(r"(?:export|cxport|expart|born|cxpert|expcrt)\s+(?:scheduled|acheduled|schedu[a-z]*|sch[\w]+)\b", text, re.IGNORECASE)
    if m:
        print("MATCH ANCHOR:", m.group(0))
    else:
        print("NO ANCHOR MATCH")

if __name__ == "__main__":
    text = open("static/debug/ocr_fullpage_p2.txt").read()
    test_chassis(text)
    test_expiry(text)
