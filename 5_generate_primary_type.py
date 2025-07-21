import argparse
from typing import Dict, Set
from pathlib import Path
import logging
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

class PrimaryConstraintGenerator:
    def __init__(self, model_name_or_path: str, tp: int = 1):
        self.model = LLM(
            model=model_name_or_path,
            tokenizer=model_name_or_path,
            tensor_parallel_size=tp,
            dtype="auto",
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.sampling_params = SamplingParams(
            max_tokens=4096,
            temperature=0.7,
            top_p=0.8,
            top_k=20,
        )
        self.primary_types = {}  # 改为字典存储，保留描述 {type_name: description}

    def _build_prompt(self, num_to_generate: int = 10) -> str:
        prompt = f"""Generate {num_to_generate} high-level primary constraint types for text generation responses.

Requirements:
1. Focus on COMMON user needs (exclude rare/niche requirements)
2. Categories should be MUTUALLY EXCLUSIVE
3. Each type should represent a DISTINCT dimension of constraints
4. Use broad categories like 'Content', 'Format' etc.

Output format:
• [Type Name]: [Description (1 sentence)]

Examples:
• Content Constraints: Requirements about the substance and information included
• Format Constraints: Requirements about the structure and presentation
• Style Constraints: Requirements about linguistic style and tone
• Example Constraints: Requirements about following a limited set of samples"""
        messages = [{"role": "user", "content": prompt}]
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

    def _parse_output(self, text: str) -> Dict[str, str]:
        types = {}
        for line in text.split('\n'):
            line = line.strip()
            if line.startswith('• ') or line.startswith('- '):
                parts = line[2:].split(':', 1)  # 只分割第一个冒号
                if len(parts) == 2:
                    type_name = parts[0].strip()
                    description = parts[1].strip()
                    if type_name:
                        types[type_name] = description
        return types

    def generate(self, output_file: str, num_prompts: int = 3):
        # Build all prompts at once
        prompts = [self._build_prompt() for _ in range(num_prompts)]
        
        # Generate all outputs in a single batch
        logger.info(f"Generating {num_prompts} batches in a single pass...")
        outputs = self.model.generate(prompts, self.sampling_params)
        
        # Process all outputs
        total_added = 0
        for output in outputs:
            generated_text = output.outputs[0].text
            new_types = self._parse_output(generated_text)
            added = {k: v for k, v in new_types.items() if k not in self.primary_types}
            self.primary_types.update(added)
            total_added += len(added)
            logger.info(f"Added {len(added)} new types from this batch")

        # Save results with descriptions
        with open(output_file, 'w', encoding='utf-8') as f:
            for type_name, description in sorted(self.primary_types.items()):
                f.write(f"{type_name}: {description}\n")  # 保持原始格式
        logger.info(f"Generated {len(self.primary_types)} unique primary constraint types (added {total_added} total)")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--tp", type=int, default=8)
    parser.add_argument("--num_prompts", type=int, default=1000,
                       help="Number of prompt variations to generate")
    args = parser.parse_args()
    
    generator = PrimaryConstraintGenerator(args.model, args.tp)
    generator.generate(args.output, args.num_prompts)

if __name__ == "__main__":
    main()