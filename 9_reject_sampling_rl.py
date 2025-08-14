import json
import concurrent.futures
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
import random

def execute_evaluation_function(func_code: str, test_response: str = "default test string", timeout: int = 5) -> bool:
    """执行评估函数，增加超时机制"""
    try:
        exec_globals = {}
        exec(func_code, exec_globals)
        
        for name, obj in exec_globals.items():
            if callable(obj) and not name.startswith('__'):
                with ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(obj, test_response)
                    try:
                        result = future.result(timeout=timeout)
                        if isinstance(result, bool):
                            return True  # 只要函数能执行完成就返回True，不关心实际结果
                        elif isinstance(result, (int, float)):
                            return True
                        return True  # 默认返回True，只要不报错
                    except concurrent.futures.TimeoutError:
                        print(f"Function {name} timed out after {timeout} seconds")
                        return False
                    except Exception as e:
                        print(f"Error calling {name}: {e}")
                        return False
        return False
    except Exception as e:
        print(f"Evaluation function error: {e}")
        return False

def filter_rl_data(input_file: str, output_file: str) -> None:
    """
    筛选RL数据：
    1. 过滤掉questions或functions为空的条目
    2. 检查所有function代码是否能在5秒内无error执行
    3. 保存符合条件的数据条目
    """
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    filtered_data = []
    
    for item in data:
        functions = item.get("functions", [])
        questions = item.get("questions", [])
        
        # 过滤掉questions或functions为空的条目
        if not functions or not questions:
            print(f"Skipping item with empty functions or questions: {item.get('id', 'unknown')}")
            continue
            
        # 检查所有function代码
        all_functions_valid = True
        with ThreadPoolExecutor(max_workers=16) as executor:
            futures = {executor.submit(execute_evaluation_function, func): func for func in functions}
            for future in as_completed(futures):
                try:
                    if not future.result():
                        all_functions_valid = False
                        break
                except Exception as e:
                    print(f"Function validation failed: {e}")
                    all_functions_valid = False
                    break
        
        # 如果所有function都有效，则保留该条目
        if all_functions_valid:
            filtered_data.append(item)

    # 保存结果
    print(f"Total valid RL data items: {len(filtered_data)}")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(filtered_data, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    input_file = "data/instructions.json"
    output_file = "data/filtered_responses_rl.json"
    filter_rl_data(input_file, output_file)