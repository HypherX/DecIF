import argparse
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Any
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

class InstructionConflictDetector:
    """A class for detecting and resolving conflicts in instructions."""
    
    def __init__(
        self,
        model_name_or_path: str,
        tp: int = 1,
        max_tokens: int = 4096,
        temperature: float = 0.6,
    ):
        """Initialize the detector with model configuration."""
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

    def detect_and_refine_instructions(
        self,
        input_file: str,
        output_file: str,
    ) -> None:
        """Detect conflicts in instructions and refine them if necessary."""
        try:
            # Load queries
            data = self._load_queries(input_file)
            logger.info(f"Loaded {len(data)} queries from {input_file}")
            
            # Extract instructions
            instructions = [item["instruction"] for item in data]
            
            # Prepare prompts for conflict detection
            prompts = self._prepare_prompts(instructions)
            
            # Batch generate all responses
            logger.info(f"Processing {len(instructions)} instructions for conflict detection...")
            outputs = self.llm_engine.generate(prompts, self.sampling_params)
            
            # Process results into refined instructions or original instruction
            for idx, (item, output) in enumerate(zip(data, outputs)):
                try:
                    response = output.outputs[0].text.strip()
                    
                    # Parse the response into structured format
                    original = item["instruction"]
                    conflict = False
                    refined = original
                    
                    if "- Conflict: True" in response:
                        conflict = True
                        refined_start = response.find("- Refined: ")
                        if refined_start != -1:
                            refined = response[refined_start + len("- Refined: "):].strip()
                    
                    # Add results to the original data
                    item["has_conflict"] = conflict
                    item["refined_instruction"] = refined
                
                except Exception as e:
                    logger.error(f"Error processing output for instruction: {item['instruction'][:50]}... Error: {str(e)}")
                    item["has_conflict"] = False
                    item["refined_instruction"] = item["instruction"]
            
            # Save results
            self._save_results(data, output_file)
            
            logger.info(f"Successfully processed instructions and saved to {output_file}")
            
        except Exception as e:
            logger.error(f"Error in instruction processing: {str(e)}")
            raise

    def _load_queries(self, input_file: str) -> List[Dict[str, Any]]:
        """Load instructions from JSON file."""
        try:
            with open(input_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return data
        except Exception as e:
            logger.error(f"Error loading queries from {input_file}: {str(e)}")
            raise

    def _prepare_prompts(self, instructions: List[str]) -> List[str]:
        """Prepare prompts for conflict detection."""
        template = (
            "You are an expert in analyzing instructions for internal conflicts. "
            "Your task is to analyze the following instruction:\n\n"
            "{instruction}\n\n"
            "Follow these steps:\n"
            "1. Check if there are any conflicting requirements (e.g., requiring both Chinese and English).\n"
            "2. If there is a conflict, refine the instruction to resolve it. The refined instruction must be clear, concise, and free of any explanatory text.\n"
            "3. If there is no conflict, return the original instruction unchanged.\n"
            "4. Format your response as follows:\n"
            "- Original: <original_instruction>\n"
            "- Conflict: True/False\n"
            "- Refined: <refined_instruction>\n"
            "Ensure that the 'Refined' field contains ONLY the refined instruction without any additional explanations or context."
        )
        
        prompts = []
        for instruction in instructions:
            prompt = template.format(instruction=instruction)
            try:
                chat_message = self.tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False
                )
                prompts.append(chat_message)
            except Exception as e:
                logger.error(f"Error applying chat template: {str(e)}")
                continue
                
        return prompts

    def _save_results(
        self,
        data: List[Dict[str, Any]],
        output_file: str,
    ) -> None:
        """Save results in JSON format."""
        try:
            Path(output_file).parent.mkdir(parents=True, exist_ok=True)
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except IOError as e:
            logger.error(f"Error saving results to {output_file}: {str(e)}")
            raise


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Detect and resolve conflicts in instructions using vLLM."
    )
    
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        required=True,
        help="Path or name of the model to load",
    )
    parser.add_argument(
        "--input_file",
        type=str,
        required=True,
        help="Path to JSON file with instructions",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="Path to save refined instructions",
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
        
        # Initialize detector
        detector = InstructionConflictDetector(
            model_name_or_path=args.model_name_or_path,
            tp=args.tp,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
        )
        
        # Detect and refine instructions
        detector.detect_and_refine_instructions(
            input_file=args.input_file,
            output_file=args.output_file,
        )
        
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        raise


if __name__ == "__main__":
    main()