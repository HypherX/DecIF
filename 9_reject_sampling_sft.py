import json
import concurrent.futures
from typing import List, Dict, Any, Tuple
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

def execute_evaluation_function(func_code, response, timeout=5):
    """执行评估函数，增加超时机制"""
    try:
        exec_globals = {}
        exec(func_code, exec_globals)
        
        for name, obj in exec_globals.items():
            if callable(obj) and not name.startswith('__'):
                with ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(obj, response)
                    try:
                        result = future.result(timeout=timeout)
                        if isinstance(result, bool):
                            return result
                        elif isinstance(result, (int, float)):
                            return bool(result)
                        return False
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

def filter_data_with_evaluation(input_file):
    """过滤数据，只保留完全通过验证的responses"""
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    filtered_data = []
    
    for item in data:
        responses = item.get("responses", [])
        functions = item.get("functions", [])
        questions = item.get("questions", [])
        
        if not responses:
            continue
            
        # 检查每个response是否通过所有验证
        passing_responses = []
        for response in responses:
            # 验证functions
            func_passed = True
            if functions:
                with ThreadPoolExecutor(max_workers=256) as executor:
                    futures = {executor.submit(execute_evaluation_function, func, response): func for func in functions}
                    for future in as_completed(futures):
                        if not future.result():
                            func_passed = False
                            break
            
            # 如果functions验证失败，跳过questions验证
            if not func_passed:
                continue
                
            # 验证questions
            ques_passed = True
            if questions:
                # 这里需要后续LLM验证
                pass
            
            if func_passed and (not questions or ques_passed):
                passing_responses.append(response)
                
        # 如果有通过验证的responses，则保留该item
        if passing_responses:
            new_item = item.copy()
            new_item["responses"] = passing_responses  # 只保留通过验证的responses
            filtered_data.append(new_item)

    print(f"Total filtered data: {len(filtered_data)}")
    return filtered_data

def filter_data_with_questions(
    data: List[Dict],
    llm: LLM,
    tokenizer,
    sampling_params,
) -> List[Dict]:
    """进行question验证，只保留完全通过验证的responses"""
    prompts, index_map = generate_verification_prompts(data, tokenizer)
    if not prompts:
        print("No prompts generated for questions verification.")
        return data

    print(f"Total verification prompts: {len(prompts)}")
    outputs = llm.generate(prompts, sampling_params)

    # 记录每个response的question验证结果
    response_results = {}
    for (idx, resp_idx), output in zip(index_map, outputs):
        content = output.outputs[0].text
        is_pass = parse_yes_no(content)
        
        if idx not in response_results:
            response_results[idx] = {}
        if resp_idx not in response_results[idx]:
            response_results[idx][resp_idx] = []
            
        response_results[idx][resp_idx].append(is_pass)

    # 筛选数据：只保留完全通过question验证的responses
    new_data = []
    for idx, item in enumerate(data):
        if "questions" not in item or not item["questions"]:
            new_data.append(item)
            continue
            
        # 收集通过所有question验证的responses
        passing_responses = []
        if idx in response_results:
            for resp_idx, ques_results in response_results[idx].items():
                if all(ques_results):
                    passing_responses.append(item["responses"][resp_idx])
                    
        if passing_responses:
            # 创建新item，只保留通过验证的responses
            new_item = item.copy()
            new_item["responses"] = passing_responses
            new_data.append(new_item)

    print(f"After question filtering: {len(new_data)} items retained.")
    return new_data
def setup_vllm(model_path: str, tp: int = 8):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    sampling_params = SamplingParams(
        max_tokens=10,
        temperature=0.6,
        top_p=0.95,
    )
    llm = LLM(
        model=model_path,
        tokenizer=model_path,
        tokenizer_mode="auto",
        tensor_parallel_size=tp,
        dtype="bfloat16"
    )
    return llm, tokenizer, sampling_params

def apply_chat_template(question: str, response: str, tokenizer):
    messages = [
        {"role": "user", "content": f"Verify the following response to the question:\n\n"
                                    f"Question: {question}\n"
                                    f"Response: {response}\n\n"
                                    f"Answer only 'yes' or 'no'."}
    ]
    try:
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        return text
    except Exception as e:
        print(f"Error applying chat template: {e}")
        return None

def generate_verification_prompts(data: List[Dict], tokenizer) -> Tuple[List[str], List[Tuple[int, int]]]:
    """生成验证prompts，返回prompts列表和(index, response_index)映射"""
    prompts = []
    index_map = []
    
    for idx, item in enumerate(data):
        if "questions" not in item or not item["questions"]:
            continue
            
        responses = item.get("responses", [])
        questions = item["questions"]
        
        for resp_idx, response in enumerate(responses):
            for question in questions:
                prompt = apply_chat_template(question, response, tokenizer)
                if prompt:
                    prompts.append(prompt)
                    index_map.append((idx, resp_idx))
                    
    return prompts, index_map

def parse_yes_no(output: str) -> bool:
    output = output.strip().lower()
    if "yes" in output:
        return True
    elif "no" in output:
        return False
    else:
        print(f"Warning: unexpected output '{output}', defaulting to False")
        return False

def filter_data_with_evaluation(input_file):
    """过滤数据，只保留完全通过验证的responses"""
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    filtered_data = []
    
    for item in data:
        responses = item.get("responses", [])
        functions = item.get("functions", [])
        questions = item.get("questions", [])
        
        if not responses:
            continue
            
        # 检查每个response是否通过所有验证
        passing_responses = []
        for response in responses:
            # 验证functions
            func_passed = True
            if functions:
                with ThreadPoolExecutor(max_workers=256) as executor:
                    futures = {executor.submit(execute_evaluation_function, func, response): func for func in functions}
                    for future in as_completed(futures):
                        if not future.result():
                            func_passed = False
                            break
            
            # 如果functions验证失败，跳过questions验证
            if not func_passed:
                continue
                
            # 验证questions
            ques_passed = True
            if questions:
                # 这里需要后续LLM验证
                pass
            
            if func_passed and (not questions or ques_passed):
                passing_responses.append(response)
                
        # 如果有通过验证的responses，则保留该item
        if passing_responses:
            new_item = item.copy()
            new_item["responses"] = passing_responses  # 只保留通过验证的responses
            filtered_data.append(new_item)

    print(f"Total filtered data: {len(filtered_data)}")
    return filtered_data

def filter_data_with_questions(
    data: List[Dict],
    llm: LLM,
    tokenizer,
    sampling_params,
) -> List[Dict]:
    """进行question验证，只保留完全通过验证的responses"""
    prompts, index_map = generate_verification_prompts(data, tokenizer)
    if not prompts:
        print("No prompts generated for questions verification.")
        return data

    print(f"Total verification prompts: {len(prompts)}")
    outputs = llm.generate(prompts, sampling_params)

    # 记录每个response的question验证结果
    response_results = {}
    for (idx, resp_idx), output in zip(index_map, outputs):
        content = output.outputs[0].text
        is_pass = parse_yes_no(content)
        
        if idx not in response_results:
            response_results[idx] = {}
        if resp_idx not in response_results[idx]:
            response_results[idx][resp_idx] = []
            
        response_results[idx][resp_idx].append(is_pass)

    # 筛选数据：只保留完全通过question验证的responses
    new_data = []
    for idx, item in enumerate(data):
        if "questions" not in item or not item["questions"]:
            new_data.append(item)
            continue
            
        # 收集通过所有question验证的responses
        passing_responses = []
        if idx in response_results:
            for resp_idx, ques_results in response_results[idx].items():
                if all(ques_results):
                    passing_responses.append(item["responses"][resp_idx])
                    
        if passing_responses:
            # 创建新item，只保留通过验证的responses
            new_item = item.copy()
            new_item["responses"] = passing_responses
            new_data.append(new_item)

    print(f"After question filtering: {len(new_data)} items retained.")
    return new_data

if __name__ == "__main__":
    input_file = "data/responses.json"
    output_file = "data/filtered_responses.json"
    
    # 第一步：基于function验证过滤数据
    print("\nFiltering based on functions...")
    filtered_data = filter_data_with_evaluation(input_file)
    
    # 第二步：设置vLLM
    model_path = "../models/qwen3-8b"
    tp = 8
    print("Initializing vLLM engine...")
    llm, tokenizer, sampling_params = setup_vllm(model_path, tp)
    
    # 第三步：基于question验证过滤数据
    print("Filtering based on questions...")
    final_data = filter_data_with_questions(filtered_data, llm, tokenizer, sampling_params)
    
    # 保存结果
    print(f"Saving final filtered data to: {output_file}")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(final_data, f, indent=2, ensure_ascii=False)
    
    print(f"\nProcessing completed. Total filtered items: {len(final_data)}")