import json
import random

with open("data/filtered_responses.json", "r") as f:
    data = json.load(f)

# data = random.sample(data, 10000)

results = []
for d in data:
    results.append({
        "messages": [
            {
                "role": "user",
                "content": d["instruction"]
            },
            {
                "role": "assistant",
                "content": d["responses"][0]
            }
        ]
    })

with open("../LLaMA-Factory/data/decif-20k.jsonl", "w") as f:
    for r in results:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")