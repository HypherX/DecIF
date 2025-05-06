import argparse
import json
from pathlib import Path
import logging
from typing import Dict, List, Optional, Tuple
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class VLLM_Engine:
    """A class for managing vLLM engine with configuration and generation capabilities."""

    def __init__(
        self,
        model_name_or_path: str,
        tp: int = 1,
        max_tokens: int = 512,
        temperature: float = 0.7,
    ) -> None:
        """Initialize the vLLM engine with given configuration.
        
        Args:
            model_name_or_path: Path or name of the model to load
            tp: Tensor parallel size for model distribution
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature for generation
        """
        self.model_name_or_path = model_name_or_path
        self.tp = tp
        self.max_tokens = max_tokens
        self.temperature = temperature
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
            self.engine, self.sampling_params = self._create_vllm_engine()
            logger.info("Successfully initialized VLLM engine")
        except Exception as e:
            logger.error(f"Failed to initialize VLLM engine: {str(e)}")
            raise

    def _create_vllm_engine(self) -> Tuple[LLM, SamplingParams]:
        """Create and configure the vLLM engine and sampling parameters.
        
        Returns:
            Tuple of (LLM engine, SamplingParams)
        """
        try:
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
        except Exception as e:
            logger.error(f"Error creating vLLM engine: {str(e)}")
            raise


def generate_scenarios_for_meta_requests(
    vllm_engine: VLLM_Engine,
    input_file: str,
    output_file: str,
    scenarios_per_request: int = 50,
) -> None:
    """Generate scenarios for each meta request in the input file.
    
    Args:
        vllm_engine: Initialized VLLM engine instance
        input_file: Path to input file with meta requests
        output_file: Path to save generated scenarios
        scenarios_per_request: Number of scenarios to generate per meta request
    """
    try:
        # Step 1: Load and validate input file
        meta_requests = _load_meta_requests(input_file)
        logger.info(f"Loaded {len(meta_requests)} meta requests from {input_file}")

        # Step 2: Generate prompts for all meta requests
        prompts = _build_scenario_prompts(meta_requests, scenarios_per_request)
        
        # Step 3: Prepare messages for generation with chat template
        chat_messages = _prepare_chat_messages(vllm_engine.tokenizer, prompts)
        
        # Step 4: Generate scenarios in batch
        logger.info(f"Generating scenarios for {len(meta_requests)} meta requests...")
        outputs = vllm_engine.engine.generate(chat_messages, vllm_engine.sampling_params)
        
        # Step 5: Process and save results
        results = _process_generated_scenarios(meta_requests, outputs, scenarios_per_request)
        _save_results(results, output_file)
        
        logger.info(f"Successfully generated scenarios for {len(results)} meta requests")
        logger.info(f"Results saved to {output_file}")

    except Exception as e:
        logger.error(f"Error in scenario generation: {str(e)}")
        raise


def _load_meta_requests(input_file: str) -> List[str]:
    """Load meta requests from input file.
    
    Args:
        input_file: Path to input file
        
    Returns:
        List of meta requests
    """
    try:
        with open(input_file, "r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip()]
    except IOError as e:
        logger.error(f"Error loading meta requests from {input_file}: {str(e)}")
        raise


def _build_scenario_prompts(meta_requests: List[str], scenarios_per_request: int) -> List[str]:
    """Build scenario generation prompts for each meta request.
    
    Args:
        meta_requests: List of meta requests
        scenarios_per_request: Number of scenarios to generate per request
        
    Returns:
        List of formatted prompts
    """
    scenario_prompt_template = (
        'For the meta request "{meta_request}", generate {count} diverse and realistic scenarios '
        'that would require this action in everyday life. Each scenario should:\n'
        '1) Be specific with clear context (who, what, where, why)\n'
        '2) Be from different domains (work, education, personal life, etc.)\n'
        '3) Be 1-2 sentences maximum\n'
        '4) Use hyphen formatting (- ...) for each scenario\n\n'
        'Example:\n'
        'Meta request: "create guide"\n'
        '- A fitness trainer needs to create a workout guide for elderly clients at a local community center\n'
        '- A software company wants to create an onboarding guide for new remote employees\n'
        '- A parent needs to create a morning routine guide for their children before school\n'
        '...\n\n'
        'Now generate scenarios for:\n'
        'Meta request: {meta_request}'
    )
    
    return [
        scenario_prompt_template.format(
            meta_request=request,
            count=scenarios_per_request
        )
        for request in meta_requests
    ]


def _prepare_chat_messages(tokenizer, prompts: List[str]) -> List[str]:
    """Apply chat template to prompts.
    
    Args:
        tokenizer: Tokenizer instance
        prompts: List of raw prompts
        
    Returns:
        List of formatted chat messages
    """
    try:
        return [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False
            )
            for prompt in prompts
        ]
    except Exception as e:
        logger.error(f"Error applying chat template: {str(e)}")
        raise


def _process_generated_scenarios(
    meta_requests: List[str],
    outputs: List,
    max_scenarios: int
) -> Dict[str, List[str]]:
    """Process generated outputs into structured scenarios.
    
    Args:
        meta_requests: List of original meta requests
        outputs: Generated outputs from vLLM
        max_scenarios: Maximum scenarios to keep per request
        
    Returns:
        Dictionary mapping meta requests to their scenarios
    """
    results = {}
    
    for request, output in zip(meta_requests, outputs):
        generated_text = output.outputs[0].text.strip()
        if "</think>" in generated_text:
            generated_text = generated_text.split("</think>")[1].strip()
        
        # Parse scenarios with hyphen formatting
        scenarios = []
        for line in generated_text.split('\n'):
            line = line.strip()
            if line.startswith('-'):
                scenario = line[1:].strip()
                if scenario.endswith('.'):
                    scenario = scenario[:-1].strip()
                if scenario:
                    scenarios.append(scenario)
        
        # Deduplicate and limit scenarios
        unique_scenarios = list(dict.fromkeys(scenarios))[:max_scenarios]
        results[request] = unique_scenarios
    
    return results


def _save_results(results: Dict[str, List[str]], output_file: str) -> None:
    """Save generated scenarios to JSON file.
    
    Args:
        results: Dictionary of meta requests and their scenarios
        output_file: Path to output file
    """
    try:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
    except IOError as e:
        logger.error(f"Error saving results to {output_file}: {str(e)}")
        raise


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.
    
    Returns:
        Namespace object containing parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Generate diverse scenarios for meta requests using vLLM."
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
        help="Path to input file with meta requests",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="Path to save generated scenarios (JSON format)",
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
        help="Maximum number of tokens to generate (default: 2048)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature for generation (default: 0.7)",
    )
    parser.add_argument(
        "--scenarios_per_request",
        type=int,
        default=30,
        help="Number of scenarios to generate per meta request (default: 20)",
    )
    
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    """Validate command line arguments.
    
    Args:
        args: Parsed command line arguments
        
    Raises:
        ValueError: If any argument is invalid
    """
    if args.tp < 1:
        raise ValueError("Tensor parallel size (--tp) must be at least 1")
    if args.max_tokens < 1:
        raise ValueError("Max tokens must be at least 1")
    if not 0 <= args.temperature <= 2:
        raise ValueError("Temperature must be between 0 and 2")
    if args.scenarios_per_request < 1:
        raise ValueError("Scenarios per request must be at least 1")


def main() -> None:
    """Main execution function."""
    try:
        # Parse and validate arguments
        args = parse_args()
        validate_args(args)
        
        # Initialize vLLM engine
        vllm_engine = VLLM_Engine(
            model_name_or_path=args.model_name_or_path,
            tp=args.tp,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
        )
        
        # Generate and save scenarios
        generate_scenarios_for_meta_requests(
            vllm_engine=vllm_engine,
            input_file=args.input_file,
            output_file=args.output_file,
            scenarios_per_request=args.scenarios_per_request,
        )
        
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        raise


if __name__ == "__main__":
    main()