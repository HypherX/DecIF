import json

with open("refined_data_path", "r") as f:
    data = json.load(f)

result = []
for d in data:
    if "NO" in d["evaluation_results"]:
        continue
    result.append(d)

print(len(result))

with open("output_path", "w") as f:
    json.dump(result, f, indent=4, ensure_ascii=False)