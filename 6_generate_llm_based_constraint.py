import argparse
import json
import re
from typing import Dict, List
from pathlib import Path
import logging
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

class LLMBasedConstraintGenerator:
    def __init__(self, model_name_or_path: str, tp: int = 1):
        self.model = LLM(
            model=model_name_or_path,
            tokenizer=model_name_or_path,
            tensor_parallel_size=tp,
            dtype="auto",
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.sampling_params = SamplingParams(
            max_tokens=2048,
            temperature=0.7,  # Higher temperature for creative questions
            top_p=0.9,
        )
        self.constraints = []

    def load_primary_types(self, input_file: str) -> List[str]:
        with open(input_file, 'r', encoding='utf-8') as f:
            return [line.split(':', 1)[0].strip() for line in f if line.strip()]

    def _build_prompt(self, primary_type: str) -> str:
        prompt = f"""You are an expert in evaluating language model outputs through LLM-based semantic constraints.

=== TASK ===
Generate MULTIPLE constraints for: "{primary_type}"

=== REQUIREMENTS ===
1. Each constraint should:
   - Focus on meaning, appropriateness, coherence, style, or nuance
   - Be impossible or impractical to verify using strict programmatic code
   - Require semantic or subjective judgment

2. For EACH constraint, provide:
   - Natural language description
   - A precise yes/no verification question based on the description

3. DO NOT include constraints related to:
   - JSON format
   - Character or word count
   - Keyword presence
   - Capitalization, punctuation, or grammar
   - Regex or other rule-based methods

4. If "{primary_type}" is mostly rule-verifiable or unsuitable for semantic evaluation, output:
   No constraints

=== EXAMPLE (Valid Type: Style Constraints) ===
1. Response must use a formal tone
   Question: "Is the tone consistently formal without informal expressions?"

2. Avoid biased or discriminatory language
   Question: "Does the response avoid biased or exclusionary phrasing?"

=== EXAMPLE (Invalid Type: Length Constraints) ===
Output:
No constraints

=== YOUR TASK ===
Generate LLM-based constraints for: "{primary_type}"
"""

        messages = [{"role": "user", "content": prompt}]
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )


    def generate(self, primary_file: str, output_file: str):
        primary_types = self.load_primary_types(primary_file)
        
        # Build all prompts
        prompts = []
        for pt in primary_types:
            prompts.append(self._build_prompt(pt))
        
        # Batch generate all
        logger.info(f"Generating constraints for {len(primary_types)} primary types...")
        outputs = self.model.generate(prompts, self.sampling_params)
        
        # Process all outputs
        for primary_type, output in zip(primary_types, outputs):
            self._parse_output(primary_type, output.outputs[0].text)
        
        # Save results
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump({"llm_based": self.constraints}, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved {len(self.constraints)} LLM-based constraints to {output_file}")

    def _parse_output(self, primary_type: str, text: str):
        if "No constraints" in text:
            logger.info(f"Skipping code-verifiable type: {primary_type}")
            return
            
        # Split by numbered constraints
        constraint_blocks = re.split(r'\n\d+\. ', text.strip())
        if len(constraint_blocks) <= 1:  # Fallback to dash splitting
            constraint_blocks = re.split(r'\n- ', text.strip())
        
        for block in constraint_blocks[1:]:  # Skip first split result
            if not block.strip():
                continue
                
            # Extract description and question
            desc_end = block.find('\n')
            if desc_end == -1:
                continue
                
            description = block[:desc_end].strip()
            question_match = re.search(r'Question:\s*"(.*?)"', block)
            
            if question_match:
                question = question_match.group(1).strip()
                self.constraints.append({
                    "description": description,
                    "verification_question": question
                })


def main():
    parser = argparse.ArgumentParser(description="Generate LLM-based constraints")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--primary", type=str, required=True,
                       help="File with primary constraint types")
    parser.add_argument("--output", type=str, required=True,
                       help="Output JSON file")
    parser.add_argument("--tp", type=int, default=8)
    args = parser.parse_args()
    
    generator = LLMBasedConstraintGenerator(args.model, args.tp)
    generator.generate(args.primary, args.output)

if __name__ == "__main__":
    main()