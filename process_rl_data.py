import json
import random

with open("data/filtered_responses_rl.json", "r") as f:
    data = json.load(f)

data = random.sample(data, 22001)

results = []
for d in data:
    # del d["responses"]
    # del d["evaluation_results"]
    if "Instruction:" in d["instruction"]:
        d["instruction"] = d["instruction"].split("Instruction:")[1].strip()
    results.append(d)

eval_data = [results[0]]

with open("data/train.jsonl", "w") as f:
    for r in results:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

with open("data/test.jsonl", "w") as f:
    for r in eval_data:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")