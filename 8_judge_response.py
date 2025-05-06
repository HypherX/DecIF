import argparse
import json
import logging
import re
from pathlib import Path
from typing import List, Dict, Any, Tuple
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

class ResponseEvaluator:
    """A class for evaluating a single response against criteria using vLLM."""
    
    def __init__(
        self,
        model_name_or_path: str,
        tp: int = 1,
        max_tokens: int = 4096,
        temperature: float = 0.0,  # Lower temperature for deterministic evaluations
    ):
        """Initialize the evaluator with model configuration."""
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

    def evaluate_response(
        self,
        data: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Evaluate a single response per instruction against criteria in a single batch."""
        try:
            # Prepare all prompts in one batch
            batch_prompts = []
            valid_indices = []
            
            for idx, item in enumerate(data):
                instruction = item.get('refined_instruction', '')
                response = item.get('response', '')
                criteria_list = item.get('criteria', [])
                
                if not response or not criteria_list:
                    continue
                    
                prompt = self._prepare_evaluation_prompt(
                    instruction=instruction,
                    response=response,
                    criteria_list=criteria_list
                )
                if prompt:
                    batch_prompts.append(prompt)
                    valid_indices.append(idx)
            
            if not batch_prompts:
                logger.warning("No valid prompts were generated for evaluation")
                return data
            
            logger.info(f"Evaluating {len(batch_prompts)} responses in a single batch...")
            outputs = self.llm_engine.generate(batch_prompts, self.sampling_params)
            
            # Initialize evaluation results for all items
            processed_data = [item.copy() for item in data]
            for item in processed_data:
                item['evaluation_results'] = []
            
            # Populate evaluation results
            for prompt_idx, output in enumerate(outputs):
                if prompt_idx >= len(valid_indices):
                    continue
                
                item_idx = valid_indices[prompt_idx]
                evaluation_result = self._parse_evaluation_response(output.outputs[0].text)
                
                try:
                    processed_data[item_idx]['evaluation_results'] = evaluation_result
                except (IndexError, KeyError) as e:
                    logger.warning(f"Error assigning evaluation result: {str(e)}")
                    continue
            
            return processed_data
        except Exception as e:
            logger.error(f"Error during response evaluation: {str(e)}")
            raise

    def _prepare_evaluation_prompt(
        self,
        instruction: str,
        response: str,
        criteria_list: List[str]
    ) -> str:
        """Prepare an evaluation prompt for a single instruction-response pair."""
        system_msg = {
            "role": "system",
            "content": """You are a Quality Assurance Specialist for AI responses. Your task is to rigorously evaluate 
whether a response meets all specified criteria. You must be thorough and impartial in your assessments.

Evaluation Guidelines:
1. Examine each criterion independently
2. Be strict but fair - only mark 'YES' if the response fully satisfies the criterion
3. Ignore any stylistic preferences not explicitly listed in the criteria
4. Focus exclusively on the criteria provided"""
        }
        
        criteria_text = "\n".join(f"{i+1}. {criterion}" for i, criterion in enumerate(criteria_list))
        
        user_msg = {
            "role": "user",
            "content": f"""INSTRUCTION:
{instruction}

RESPONSE TO EVALUATE:
{response}

EVALUATION CRITERIA:
{criteria_text}

YOUR TASK:
For each criterion above, output ONLY either 'YES' or 'NO' on its own line, in order.
The evaluation must contain exactly {len(criteria_list)} lines, one for each criterion.

EXAMPLE OUTPUT FOR 3 CRITERIA:
YES
NO
YES

BEGIN EVALUATION:"""
        }
        
        try:
            return self.tokenizer.apply_chat_template(
                [system_msg, user_msg],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except Exception as e:
            logger.warning(f"Error creating evaluation prompt: {str(e)}")
            return ""

    def _parse_evaluation_response(self, response: str) -> List[str]:
        """Parse the evaluation response into a list of YES/NO results."""
        # Normalize the response
        response = response.strip().upper()
        
        # Extract all YES/NO matches in order
        evaluations = re.findall(r'\b(YES|NO)\b', response)
        
        # Validate we got at least one evaluation
        if not evaluations:
            logger.warning(f"No valid evaluations found in response: {response[:200]}...")
            return []
            
        return evaluations


def load_data(input_path: Path) -> List[Dict[str, Any]]:
    """Load the input JSON data."""
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
    """Save the processed data to JSON file."""
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
    evaluator: ResponseEvaluator,
) -> List[Dict[str, Any]]:
    """Process all data with single response evaluation."""
    try:
        # Batch evaluate all responses at once
        logger.info("Evaluating responses against criteria...")
        processed_data = evaluator.evaluate_response(original_data)
        
        # Validate evaluations were processed
        for item in processed_data:
            if 'response' in item and 'evaluation_results' in item:
                criteria_count = len(item.get('criteria', []))
                eval_count = len(item['evaluation_results'])
                if criteria_count != eval_count:
                    logger.warning(f"Mismatch in criteria/evaluation counts for item: {item.get('instruction', '')[:50]}...")
        
        return processed_data
    except Exception as e:
        logger.error(f"Error during data processing: {str(e)}")
        raise


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate a single response against criteria."
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
        help="Sampling temperature (default: 0.0 for deterministic evaluations)",
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
        
        # Initialize evaluator
        evaluator = ResponseEvaluator(
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
        processed_data = process_data(original_data, evaluator)
        
        # Save results
        logger.info(f"Saving results to {args.output_file}...")
        save_results(processed_data, args.output_file)
        
        logger.info("Processing completed successfully")
        
    except Exception as e:
        logger.error(f"Script execution failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()