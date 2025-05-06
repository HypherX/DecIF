import argparse
import json
import random
import logging
from pathlib import Path
from typing import Dict, List, Tuple
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


CONSTRAINT_TYPES = {
    "Include Keywords": {
        "weight": 1,
        "description": "Include specific keyword(s) in the instruction's response."
    },
    "Keyword Frequency": {
        "weight": 1,
        "description": "In the instruction's response, the specific keywords should appear specific times."
    },
    "Forbidden Words": {
        "weight": 1,
        "description": "Do not include keyword(s) in the instruction's response."
    },
    "Letter Frequency": {
        "weight": 1,
        "description": "In the instruction's response, the specific letter should appear specific times."
    },
    "Response Language": {
        "weight": 1,
        "description": "The instruction's response should be in specific language, no other language is allowed."
    },
    "Number Paragraphs": {
        "weight": 1,
        "description": "The instruction's response should contain specific paragraphs and separate paragraphs using specific symbol"
    },
    "Number Words": {
        "weight": 1,
        "description": "The instruction's response should be at least / around / at most specific words."
    },
    "Number Sentences": {
        "weight": 1,
        "description": "The instruction's response should be with at least / around / at most specific sentences."
    },
    "mixed": {
        "weight": 1,
        "description": "The instruction's response should be specific paragraphs. Paragraphs and only paragraphs are separated with each other by two line breaks. The specific paragraph must start with specific word or phrase."
    },
    "Postscript": {
        "weight": 1,
        "description": "The instruction's response should explicitly add a postscript starting with specific postscript markers at the end."
    },
    "Number Placeholder": {
        "weight": 1,
        "description": "The instruction's response should contain specific placeholders represented by square brackets."
    },
    "Number Bullets": {
        "weight": 1,
        "description": "The instruction's response should contain specific bullet points."
    },
    "Title": {
        "weight": 1,
        "description": "The instruction's response should contain a title in specific forms."
    },
    "Choose From": {
        "weight": 1,
        "description": "The instruction's response should be with one of the specific options"
    },
    "Minimum Number Highlighted Section": {
        "weight": 1,
        "description": "Highlight specific sections in the instruction's response with specific forms."
    },
    "Multiple Sections": {
        "weight": 1,
        "description": "The instruction's response should have specific sections. Mark the beginning of each section with specific section splitter."
    },
    "Multiple Format": {
        "weight": 1,
        "description": "The instruction's response should be in JSON, Table, HTML, XML, LaTeX, Markdown format."
    },
    "Repeat Prompt": {
        "weight": 1,
        "description": "First, repeat the request without change in the instruction's response, then give the answer (do not say anything before repeating the request; the request you need to repeat does not include this sentence)."
    },
    "Two Responses": {
        "weight": 1,
        "description": "Give two different responses in the instruction's response. Responses and only responses should be separated by specific symbols."
    },
    "All Uppercase": {
        "weight": 1,
        "description": "The instruction's response should contain capital letters only."
    },
    "All Lowercase": {
        "weight": 1,
        "description": "The instruction's response should contain lowercase letters only."
    },
    "Frequency of Allcapital Words": {
        "weight": 1,
        "description": "In the instruction's response, words with all capital letters should appear at least / around / at most specific times."
    },
    "End Checker": {
        "weight": 1,
        "description": "The instruction's response should end with specific phrase. No other words should follow this phrase."
    },
    "Start Checker": {
        "weight": 1,
        "description": "The instruction's response should start with specific phrase."
    },
    "Quotation": {
        "weight": 1,
        "description": "The instruction's response should be wrapped with specific marks."
    },
    "No Commas": {
        "weight": 1,
        "description": "The instruction's response should refrain from the use of any commas."
    },
    "role-based": {
        "weight": 1,
        "description": "The instruction's response should simulate characters based on context, emulating their traits, language, and behaviors."
    },
    "scenario-based": {
        "weight": 1,
        "description": "The instruction's response should be in a specific situational demands."
    },
    "scenario-based": {
        "weight": 1,
        "description": "The instruction's response should be in a specific situational demands."
    },
    "style": {
        "weight": 1,
        "description": "The instruction's response should be in a specific style, tone or emotion."
    },
    "audience": {
        "weight": 1,
        "description": "The instruction's response should be tailored to specific audiences."
    },
}

class InstructionGenerator:
    """A class for generating verifiable instructions with constraints."""
    
    def __init__(
        self,
        model_name_or_path: str,
        tp: int = 1,
        max_tokens: int = 4096,
        temperature: float = 0.7,
    ):
        """Initialize the generator with model configuration."""
        self.model_name_or_path = model_name_or_path
        self.tp = tp
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.llm_engine, self.sampling_params = self._initialize_engine()
        
    def _initialize_engine(self) -> Tuple[LLM, SamplingParams]:
        """Initialize the vLLM engine and tokenizer."""
        try:
            logger.info("Initializing vLLM engine...")
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
            )
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path)
            return llm_engine, sampling_params
        except Exception as e:
            logger.error(f"Failed to initialize vLLM engine: {str(e)}")
            raise

    def generate_instructions(
        self,
        persona_data: Dict[str, List[str]],
        output_file: str,
        total_num: int = 30000,
    ) -> None:
        """Generate and save instructions based on personas."""
        try:
            # Sample personas and prepare prompts
            samples, prompts = self._prepare_prompts(persona_data, total_num)
            
            # Batch generate all instructions
            logger.info(f"Generating {len(prompts)} instructions in batch...")
            outputs = self.llm_engine.generate(prompts, self.sampling_params)
            
            # Process outputs and save results
            self._process_and_save_results(samples, outputs, output_file)
            
            logger.info(f"Successfully generated instructions and saved to {output_file}")
            
        except Exception as e:
            logger.error(f"Error in instruction generation: {str(e)}")
            raise

    def _prepare_prompts(
        self,
        persona_data: Dict[str, List[str]],
        total_num: int,
    ) -> Tuple[List[Dict], List[str]]:
        """Prepare samples and prompts for generation."""
        samples = self._sample_personas(persona_data, total_num)
        prompts = []
        
        for sample in samples:
            prompt = self._generate_prompt(
                sample['persona'],
                sample['request'],
                sample['constraints']
            )
            messages = [{"role": "user", "content": prompt}]
            
            try:
                chat_message = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False
                )
                prompts.append(chat_message)
            except Exception as e:
                logger.error(f"Error applying chat template: {str(e)}")
                continue
                
        return samples, prompts

    def _sample_personas(
        self,
        persona_data: Dict[str, List[str]],
        total_num: int,
    ) -> List[Dict]:
        """Randomly sample personas from the loaded data."""
        all_personas = []
        for request, personas in persona_data.items():
            for persona in personas:
                all_personas.append({
                    "request": request,
                    "persona": persona,
                    "constraints": self._select_constraints()
                })
        
        if total_num > len(all_personas):
            logger.warning(
                f"Requested {total_num} samples but only {len(all_personas)} available. "
                f"Using all available samples."
            )
            total_num = len(all_personas)
            
        return random.sample(all_personas, total_num)

    def _select_constraints(self) -> List[str]:
        """Randomly select 1-5 constraint descriptions with weighted probability."""
        num_constraints = random.choices(
            [1, 2, 3, 4, 5],
            weights=[0.2, 0.3, 0.3, 0.1, 0.1],
            k=1
        )[0]
        
        weights = [details['weight'] for details in CONSTRAINT_TYPES.values()]
        descriptions = [details['description'] for details in CONSTRAINT_TYPES.values()]
        
        selected = random.choices(descriptions, weights=weights, k=num_constraints)
        return list(set(selected))

    def _generate_prompt(
        self,
        persona: str,
        request: str,
        constraints: List[str],
    ) -> str:
        """Generate the prompt for creating verifiable instructions."""
        constraints_str = "\n".join([f"- {c}" for c in constraints])
        example = "Write down the names of two famous international badminton mixed doubles tournaments and your answer should be all in capital words."
        
        return f"""
Create a verifiable instruction that the following persona might ask you to do:
{persona}

An example:
{example}

Note:
1. The above example is not tied to any particular persona, but you should create one that is unique and specific to the given persona.
2. The instruction should contain all the following verifiable constraint(s):
{constraints_str}
3. Your output should start with "User instruction:". Your output should not include an answer to the instruction.
"""

    def _process_and_save_results(
        self,
        samples: List[Dict],
        outputs: List,
        output_file: str,
    ) -> None:
        """Process outputs and save results to file."""
        for sample, output in zip(samples, outputs):
            generated_text = output.outputs[0].text.strip()
            instruction = self._process_instruction(generated_text)
            if instruction:
                sample['instruction'] = instruction
        
        # Save only samples with valid instructions
        valid_samples = [s for s in samples if 'instruction' in s]
        
        try:
            Path(output_file).parent.mkdir(parents=True, exist_ok=True)
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(valid_samples, f, indent=2, ensure_ascii=False)
        except IOError as e:
            logger.error(f"Error saving results: {str(e)}")
            raise

    def _process_instruction(self, generated_text: str) -> str:
        """Process the raw generated text to extract the instruction."""
        if '</think>' in generated_text:
            generated_text = generated_text.split("</think>")[1].strip()
        
        if generated_text.startswith("User instruction:"):
            return generated_text[len("User instruction:"):].strip()
        
        return ""


def load_personas(json_file: str) -> Dict[str, List[str]]:
    """Load personas from JSON file."""
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading personas from {json_file}: {str(e)}")
        raise


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate verifiable instructions with constraints."
    )
    
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        required=True,
        help="Path or name of the model to load",
    )
    parser.add_argument(
        "--input_json",
        type=str,
        required=True,
        help="Path to JSON file with personas",
    )
    parser.add_argument(
        "--output_json",
        type=str,
        required=True,
        help="Path to save generated instructions",
    )
    parser.add_argument(
        "--total_num",
        type=int,
        default=100,
        help="Total number of instructions to generate (default: 100)",
    )
    parser.add_argument(
        "--tp",
        type=int,
        default=4,
        help="Tensor parallel size for model distribution (default: 4)",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=4096,
        help="Maximum number of tokens to generate (default: 4096)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.6,
        help="Sampling temperature for generation (default: 0.6)",
    )
    
    return parser.parse_args()


def main() -> None:
    """Main execution function."""
    try:
        args = parse_args()
        
        # Initialize generator
        generator = InstructionGenerator(
            model_name_or_path=args.model_name_or_path,
            tp=args.tp,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
        )
        
        # Load personas
        logger.info(f"Loading personas from {args.input_json}...")
        persona_data = load_personas(args.input_json)
        
        # Generate instructions
        logger.info(f"Generating {args.total_num} instructions...")
        generator.generate_instructions(
            persona_data=persona_data,
            output_file=args.output_json,
            total_num=args.total_num,
        )
        
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        raise


if __name__ == "__main__":
    main()