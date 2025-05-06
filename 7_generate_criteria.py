import argparse
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import re

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

class CriteriaExtractor:
    """A class for extracting granular criteria from instructions using vLLM."""
    
    def __init__(
        self,
        model_name_or_path: str,
        tp: int = 1,
        max_tokens: int = 4096,
        temperature: float = 0.6,
    ):
        """Initialize the extractor with model configuration."""
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
                top_p=0.8,
                top_k=20,
            )
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path)
            return llm_engine, sampling_params
        except Exception as e:
            logger.error(f"Failed to initialize vLLM engine: {str(e)}")
            raise

    def extract_criteria(
        self,
        instructions: List[str],
    ) -> List[List[str]]:
        """Extract granular criteria from instructions in a single batch."""
        try:
            # Prepare all prompts in one batch
            batch_prompts = self._prepare_extraction_prompts(instructions)
            
            logger.info(f"Processing {len(batch_prompts)} instructions in a single batch...")
            outputs = self.llm_engine.generate(batch_prompts, self.sampling_params)
            
            # Process all responses
            all_criteria = []
            for output in outputs:
                response = output.outputs[0].text.strip()
                criteria_list = self._parse_criteria_response(response)
                all_criteria.append(criteria_list)
            
            return all_criteria
        except Exception as e:
            logger.error(f"Error during criteria extraction: {str(e)}")
            raise

    def _prepare_extraction_prompts(self, instructions: List[str]) -> List[str]:
        """Prepare all prompts for batch processing."""
        system_msg = {
            "role": "system",
            "content": "You are an expert at breaking down instructions into granular evaluation criteria."
        }
        
        prompts = []
        for instruction in instructions:
            user_msg = {
                "role": "user",
                "content": f"""You are now an Evaluation Criteria Designer , tasked with breaking down complex instructions into granular evaluation questions. These questions will be used to assess whether a response meets the requirements of the given instruction.

Your task is as follows:

Analyze the provided instruction and identify all atomic-level requirements including: content specifications, formatting constraints, stylistic guidelines, factual accuracy checks, logical consistency requirements, and any other explicit or implicit conditions that must be satisfied.
For each requirement or constraint, create a clear, concise evaluation question that can be answered with a simple "yes" or "no."
Ensure the questions are specific, actionable, and free of any explanations or additional context.
Instruction:
{instruction}

Output Format:

Each evaluation question should be on a new line.
Questions must be phrased in a way that allows for a binary ("yes" or "no") answer.
Example Output:
Does the response include at least three sources?
Is the response under 100 words?

Now, proceed with the breakdown. """
            }
            
            try:
                chat_prompt = self.tokenizer.apply_chat_template(
                    [system_msg, user_msg],
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
                prompts.append(chat_prompt)
            except Exception as e:
                logger.warning(f"Error creating prompt for instruction: {str(e)}")
                continue
                
        return prompts

    def _parse_criteria_response(self, response: str) -> List[str]:
        """Parse the criteria response into a list of strings."""
        # Split by newlines and clean each line
        criteria = [line.strip() for line in response.split('\n') if line.strip()]
        
        # Remove any numbering or bullets if present
        cleaned_criteria = []
        for criterion in criteria:
            # Remove patterns like "1. ", "- ", "* ", etc.
            cleaned = re.sub(r'^(\d+\.\s*|[-*]\s*)', '', criterion)
            cleaned_criteria.append(cleaned)
        
        return cleaned_criteria


def load_data(input_path: Path) -> List[Dict[str, Any]]:
    """Load the input JSON data while preserving all original fields."""
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if not isinstance(data, list):
                raise ValueError("Input JSON should be an array of objects")
            return data
    except Exception as e:
        logger.error(f"Error loading data from {input_path}: {str(e)}")
        raise


def save_results(data: List[Dict[str, Any]], output_path: Path) -> None:
    """Save the processed data to JSON file while preserving all original fields."""
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved {len(data)} items to {output_path}")
    except Exception as e:
        logger.error(f"Error saving results to {output_path}: {str(e)}")
        raise


def process_data(
    original_data: List[Dict[str, Any]],
    extractor: CriteriaExtractor,
) -> List[Dict[str, Any]]:
    """Process all data while preserving original structure with batch processing."""
    try:
        # Extract all instructions for batch processing
        instructions = [item.get('refined_instruction', '') for item in original_data]
        
        # Batch process all instructions at once
        logger.info("Extracting criteria from instructions...")
        all_criteria = extractor.extract_criteria(instructions)
        
        # Add criteria to each item under 'criteria' key
        processed_data = []
        for item, criteria_list in zip(original_data, all_criteria):
            new_item = item.copy()
            new_item['criteria'] = criteria_list
            processed_data.append(new_item)
        
        return processed_data
    except Exception as e:
        logger.error(f"Error during data processing: {str(e)}")
        raise


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Extract granular evaluation criteria from instructions."
    )
    
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the model directory",
    )
    parser.add_argument(
        "--input_file",
        type=Path,
        required=True,
        help="Path to input JSON file",
    )
    parser.add_argument(
        "--output_file",
        type=Path,
        required=True,
        help="Path to output JSON file",
    )
    parser.add_argument(
        "--tp",
        type=int,
        default=4,
        help="Tensor parallel size (default: 4)",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=4096,
        help="Maximum tokens to generate (default: 4096)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature (default: 0.6)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of items to process (for testing)",
    )
    
    return parser.parse_args()


def main() -> None:
    """Main execution function."""
    try:
        args = parse_args()
        
        # Initialize extractor
        extractor = CriteriaExtractor(
            model_name_or_path=args.model_path,
            tp=args.tp,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
        )
        
        # Load data
        logger.info(f"Loading data from {args.input_file}...")
        original_data = load_data(args.input_file)
        if args.limit:
            original_data = original_data[:args.limit]
        logger.info(f"Loaded {len(original_data)} items")
        
        # Process data
        processed_data = process_data(original_data, extractor)
        
        # Save results
        logger.info(f"Saving results to {args.output_file}...")
        save_results(processed_data, args.output_file)
        
        logger.info("Processing completed successfully")
        
    except Exception as e:
        logger.error(f"Script execution failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()