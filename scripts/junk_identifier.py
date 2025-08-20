import os
import json
import re
from collections import Counter
import sys
sys.stdout.reconfigure(encoding='utf-8')

data_folder = "data"
chunk_pattern = re.compile(r"^(chunks|web_chunks_part_\d+)\.json$")

chunk_texts = []

for filename in os.listdir(data_folder):
    if chunk_pattern.match(filename):
        filepath = os.path.join(data_folder, filename)
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
            for chunk in data:
                text = chunk["content"].strip()
                chunk_texts.append(text)

# Count occurrences of each unique chunk
counter = Counter(chunk_texts)

# Print chunks 
for text, count in counter.most_common():
    if count > 10 and len(text) < 500:  
        print(f"Count: {count}")
        print("-" * 80)
        print(text)
        print("-" * 80)
        print()