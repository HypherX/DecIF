import json
import random
from typing import Dict, List
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import argparse


class VLLM_Engine:
    def __init__(self, model_name_or_path: str, tp: int, max_tokens: int, temperature: float):
        self.model_name_or_path = model_name_or_path
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.tp = tp
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.engine, self.sampling_params = self.create_vllm_engine()

    def create_vllm_engine(self):
        llm_engine = LLM(
            model=self.model_name_or_path,
            tokenizer=self.model_name_or_path,
            tokenizer_mode="auto",
            tensor_parallel_size=self.tp,
            dtype="bfloat16"
        )
        sampling_params = SamplingParams(
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            top_p=0.8,
            top_k=20,
        )
        return llm_engine, sampling_params


def load_personas(json_file: str) -> Dict[str, List[str]]:
    """Load personas from JSON file"""
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def sample_personas(persona_data: Dict[str, List[str]], total_num: int) -> List[Dict]:
    """Randomly sample personas from the loaded data, return all if requested number exceeds available"""
    samples = []
    all_personas = []

    # Flatten all personas into a single list with their request types
    for request, personas in persona_data.items():
        for persona in personas:
            all_personas.append({"request": request, "persona": persona})

    # If requested number exceeds available personas, return all
    if total_num > len(all_personas):
        samples = all_personas
    else:
        samples = random.sample(all_personas, total_num)

    return samples


def generate_instruction_prompt(persona: str, request: str) -> str:
    prompt = (
        "Create a high-quality instruction that the following persona would realistically ask an AI assistant:\n"
        f"Persona: {persona}\n\n"
        "Requirements for the instruction:\n"
        "1. **Clarity**: The instruction should be unambiguous\n"
        "2. **Actionability**: The instruction should be executable through text interaction\n"
        "3. **Realism**: The instruction should reflect what this persona would genuinely ask\n"
        "4. **Proper Format**: Begin with 'User instruction:' and provide nothing else\n\n"
        "Example of some good instructions (Do not repeat and overly follow the examples):\n"
        "User instruction: A modern-day hunter-gatherer from an isolated tribe sets out to gather resources for the day. He decides to collect some fruits and hunt small animals to provide for his family. In the morning, he gathers 15 wild berries from the forest. Later, he catches 3 rabbits, each weighing 2 kilograms. In the afternoon, he finds a tree with 8 ripe bananas and decides to take them all. Finally, he discovers a small patch of edible roots and digs up 5 of them, each weighing 0.5 kilograms. How many individual food items did the hunter-gatherer collect in total during his day of gathering and hunting?\n"
        "User instruction: Write a python function to take a list of integers as input and return a new list consisting of the squares of each number in the input list. Ensure that the output list maintains the same order as the input list. Input: A list of integers, e.g., `[1, 2, 3, 4]` Expected Output: A list of integers, where each integer is the square of the corresponding integer in the input list, e.g., `[1, 4, 9, 16]` Example: ```python def square_numbers(input_list): # Your code here print(square_numbers([1, 2, 3, 4])) # Output should be [1, 4, 9, 16] print(square_numbers([-1, -2, 0, 3])) # Output should be [1, 4, 0, 9] ```\n"
        "User instruction: Could you write a Between the Lions Segment about Italy chef making pizza but he missing an item that could help to add pizza.\n\n"
        "Now generate the your instruction:"
    )
    return prompt


def process_generated_instruction(generated_text: str) -> str:
    """Process the raw generated text to extract the instruction"""
    # Remove the "User instruction:" prefix if present
    if generated_text.startswith("User instruction:"):
        generated_text = generated_text[len("User instruction:"):].strip()
        return generated_text

    return ""


def generate_instructions(vllm_engine, persona_data: Dict[str, List[str]], output_file: str, total_num: int):
    """Generate instructions based on sampled personas and save results"""
    # Sample personas (with replacement if needed)
    samples = sample_personas(persona_data, total_num)

    # Prepare all prompts
    prompts = []
    for sample in samples:
        # Generate prompt
        prompt = generate_instruction_prompt(sample['persona'], sample['request'])
        messages = [{"role": "user", "content": prompt}]

        chat_message = vllm_engine.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        prompts.append(chat_message)

    # Batch generate all instructions
    outputs = vllm_engine.engine.generate(prompts, vllm_engine.sampling_params)

    # Process outputs and add instructions to samples
    for sample, output in zip(samples, outputs):
        generated_text = output.outputs[0].text.strip()
        instruction = process_generated_instruction(generated_text)
        if not instruction:
            continue
        sample['instruction'] = instruction

    # Save results
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(samples, f, indent=2, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic instructions using vLLM engine.")
    
    parser.add_argument("--model-path", type=str, required=True,
                        help="Path to the pre-trained model")
    parser.add_argument("--tp", type=int, default=1,
                        help="Tensor parallelism size")
    parser.add_argument("--max-tokens", type=int, default=4096,
                        help="Max tokens to generate per instruction")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Sampling temperature")
    parser.add_argument("--input-json", type=str, required=True,
                        help="Path to input JSON containing personas")
    parser.add_argument("--output-json", type=str, required=True,
                        help="Path to output JSON to save generated instructions")
    parser.add_argument("--total-num", type=int, default=100,
                        help="Total number of instructions to generate")

    args = parser.parse_args()

    # Initialize engine
    print(f"Initializing vLLM engine with model: {args.model_path}")
    vllm_engine = VLLM_Engine(
        model_name_or_path=args.model_path,
        tp=args.tp,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
    )

    # Load personas
    print(f"Loading personas from {args.input_json}...")
    persona_data = load_personas(args.input_json)

    # Generate instructions
    print(f"Generating {args.total_num} instructions...")
    generate_instructions(vllm_engine, persona_data, args.output_json, args.total_num)
    print(f"Saved generated instructions to {args.output_json}")


if __name__ == "__main__":
    main()