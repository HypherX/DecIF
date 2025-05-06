import argparse
import json
import logging
from pathlib import Path
from typing import List, Dict, Tuple
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

class ResponseGenerator:
    """A class for generating responses to instructions using vLLM."""
    
    def __init__(
        self,
        model_name_or_path: str,
        tp: int = 1,
        max_tokens: int = 4096,
        temperature: float = 0.6,
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
                top_p=0.8,
                top_k=20,
            )
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path)
            return llm_engine, sampling_params
        except Exception as e:
            logger.error(f"Failed to initialize vLLM engine: {str(e)}")
            raise

    def generate_responses(
        self,
        input_json: str,
        output_json: str,
    ) -> None:
        """Generate responses for each instruction."""
        try:
            # Load queries
            queries = self._load_queries(input_json)
            logger.info(f"Loaded {len(queries)} queries from {input_json}")
            
            # Prepare all prompts
            prompts = self._prepare_prompts(queries)
            
            # Batch generate all responses
            logger.info(f"Generating responses for {len(prompts)} instructions...")
            outputs = self.llm_engine.generate(prompts, self.sampling_params)
            
            # Process results into a list of responses
            responses = self._process_outputs(outputs)
            
            # Save results
            self._save_results(queries, responses, output_json)
            
            logger.info(f"Successfully generated responses and saved to {output_json}")
            
        except Exception as e:
            logger.error(f"Error in response generation: {str(e)}")
            raise

    def _load_queries(self, input_json: str) -> List[str]:
        """Load queries from JSON file."""
        try:
            with open(input_json, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return [item["refined_instruction"].strip() for item in data if "refined_instruction" in item]
        except Exception as e:
            logger.error(f"Error loading queries from {input_json}: {str(e)}")
            raise

    def _prepare_prompts(self, queries: List[str]) -> List[str]:
        """Prepare prompts for all queries."""
        ## template for instruction-following data
        template = (
            "You are an expert tasked with answering the given query. "
            "Please provide a clear and concise response directly, "
            "without introductory phrases such as 'What a great question,' "
            "'Here is the answer,' or similar expressions. Focus solely on addressing the query.\n"
            "Now please answer the given query while strictly following its inside constraints.\n"
            "[Query] {}"
        )
        ## template for general-purpose data
        # template = (
        #     "You are an expert tasked with answering the given query. "
        #     "Please provide a clear and accurate response to the given query.\n"
        #     "[Query] {}"
        # )
        
        prompts = []
        for query in queries:
            message = {"role": "user", "content": template.format(query)}
            try:
                chat_message = self.tokenizer.apply_chat_template(
                    [message],
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False
                )
                prompts.append(chat_message)
            except Exception as e:
                logger.error(f"Error applying chat template: {str(e)}")
                continue
                
        return prompts

    def _process_outputs(self, outputs: List) -> List[str]:
        """Process outputs into a list of responses."""
        responses = []
        for output in outputs:
            response = output.outputs[0].text.strip()
            if '</think>' in response:
                response = response.split("</think>")[1].strip()
            responses.append(response)
        
        return responses

    def _save_results(
        self,
        queries: List[str],
        responses: List[str],
        output_json: str,
    ) -> None:
        """Save results with single response per query."""
        results = []
        for query, response in zip(queries, responses):
            results.append({
                "instruction": query,
                "response": response,
            })
        
        try:
            Path(output_json).parent.mkdir(parents=True, exist_ok=True)
            with open(output_json, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
        except IOError as e:
            logger.error(f"Error saving results to {output_json}: {str(e)}")
            raise


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate responses to instructions using vLLM."
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
        help="Path to JSON file with instructions",
    )
    parser.add_argument(
        "--output_json",
        type=str,
        required=True,
        help="Path to save generated responses",
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
        default=0.7,
        help="Sampling temperature for generation (default: 0.6)",
    )
    
    return parser.parse_args()


def main() -> None:
    """Main execution function."""
    try:
        args = parse_args()
        
        # Initialize generator
        generator = ResponseGenerator(
            model_name_or_path=args.model_name_or_path,
            tp=args.tp,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
        )
        
        # Generate responses
        generator.generate_responses(
            input_json=args.input_json,
            output_json=args.output_json,
        )
        
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        raise


if __name__ == "__main__":
    main()