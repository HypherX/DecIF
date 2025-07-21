import json
import random
from collections import defaultdict

def load_and_preprocess_data(filepath: str) -> list:
    """加载并预处理数据"""
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    results = []
    for d in data:
        # 创建新条目，只保留需要的字段
        new_item = {
            "instruction": d["instruction"],
            "functions": d.get("functions", []),
            "questions": d.get("questions", [])
        }
        
        # 处理instruction字段
        if "Instruction:" in new_item["instruction"]:
            new_item["instruction"] = new_item["instruction"].split("Instruction:")[1].strip()
        
        results.append(new_item)
    
    return results

def filter_non_empty(data: list) -> list:
    """过滤掉functions或questions为空的条目"""
    return [item for item in data 
            if item["functions"] and item["questions"]]

def analyze_combinations(data: list) -> dict:
    """分析数据中实际存在的组合情况"""
    combination_stats = defaultdict(int)
    for item in data:
        q_count = len(item["questions"])
        f_count = len(item["functions"])
        combination_stats[(q_count, f_count)] += 1
    return combination_stats

def balanced_sample(data: list, total: int) -> list:
    """基于实际存在的组合进行平衡抽样"""
    # 分析数据中的组合分布
    combination_stats = analyze_combinations(data)
    valid_combinations = list(combination_stats.keys())
    
    if not valid_combinations:
        return random.sample(data, min(total, len(data)))
    
    # 计算每种组合应该抽取的样本数
    samples_per_group = total // len(valid_combinations)
    remainder = total % len(valid_combinations)
    
    # 创建组合到数据的映射
    combination_map = defaultdict(list)
    for item in data:
        q = len(item["questions"])
        f = len(item["functions"])
        combination_map[(q, f)].append(item)
    
    sampled_data = []
    for i, combo in enumerate(valid_combinations):
        # 计算当前组合应该抽取的数量
        count = samples_per_group + (1 if i < remainder else 0)
        
        # 从对应分组中随机抽取
        available_items = combination_map.get(combo, [])
        if available_items:
            sampled_data.extend(random.sample(available_items, min(count, len(available_items))))
        else:
            print(f"Warning: No items found for combination questions={combo[0]}, functions={combo[1]}")
    
    # 如果总数不足，随机补充
    if len(sampled_data) < total:
        remaining = total - len(sampled_data)
        sampled_data.extend(random.sample(data, remaining))
    
    return sampled_data

def save_to_jsonl(data: list, filepath: str):
    """保存数据到jsonl文件"""
    with open(filepath, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    # 配置参数
    input_file = "data/filtered_responses_rl.json"  # 输入文件路径
    train_output = "data/train.jsonl"           # 训练集输出路径
    test_output = "data/test.jsonl"             # 测试集输出路径
    total_samples = 22001                       # 设置您想要的样本总数
    
    # 1. 加载并预处理数据
    print("正在加载和预处理数据...")
    processed_data = load_and_preprocess_data(input_file)
    
    # 2. 过滤掉functions或questions为空的条目
    print("\n过滤掉functions或questions为空的条目...")
    filtered_data = filter_non_empty(processed_data)
    print(f"原始数据量: {len(processed_data)}, 过滤后数据量: {len(filtered_data)}")
    
    # 3. 分析数据中的组合分布
    combination_stats = analyze_combinations(filtered_data)
    print("\n过滤后数据中存在的组合统计:")
    for (q, f), count in sorted(combination_stats.items()):
        print(f"问题数: {q}, 函数数: {f} - {count}条数据 ({count/len(filtered_data):.1%})")
    
    # 4. 进行平衡抽样
    print(f"\n正在进行平衡抽样(目标样本数: {total_samples})...")
    sampled_data = balanced_sample(filtered_data, total_samples)
    
    # 5. 分割训练集和测试集
    train_data = sampled_data
    test_data = [sampled_data[0]] if sampled_data else []
    
    # 6. 保存结果
    save_to_jsonl(train_data, train_output)
    save_to_jsonl(test_data, test_output)
    
    # 7. 打印最终统计信息
    final_stats = analyze_combinations(train_data)
    print("\n抽样结果分布:")
    for (q, f), count in sorted(final_stats.items()):
        print(f"问题数: {q}, 函数数: {f} - {count}条样本 ({count/len(train_data):.1%})")
    
    print(f"\n处理完成:")
    print(f"- 训练集: {len(train_data)} 条样本 (已保存到 {train_output})")
    print(f"- 测试集: {len(test_data)} 条样本 (已保存到 {test_output})")