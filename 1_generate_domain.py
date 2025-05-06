import argparse
import random
from typing import List, Set, Optional
from pathlib import Path
import logging
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class VLLM_DiverseGenerator:
    """A class for generating diverse domains using vLLM language models."""

    def __init__(
        self,
        model_name_or_path: str,
        tp: int = 1,
        max_tokens: int = 100,
        temperature: float = 0.7,
    ) -> None:
        """Initialize the generator with model configuration.
        
        Args:
            model_name_or_path: Path or name of the model to load
            tp: Tensor parallel size for model distribution
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature for generation
        """
        self.model = LLM(
            model=model_name_or_path,
            tokenizer=model_name_or_path,
            tensor_parallel_size=tp,
            dtype="auto",
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.sampling_params = SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=0.8,
            top_k=20,
        )
        self.generated_domains: Set[str] = set()

    def generate_domains(self, prompts: List[str]) -> List[List[str]]:
        """Generate domains from a list of prompts.
        
        Args:
            prompts: List of input prompts for generation
            
        Returns:
            List of lists containing generated domains for each prompt
        """
        try:
            outputs = self.model.generate(prompts, self.sampling_params)
            all_domains = []
            
            for output in outputs:
                generated_text = output.outputs[0].text.strip()
                if "</think>" in generated_text:
                    generated_text = generated_text.split("</think>")[1].strip()
                new_domains = self._parse_generated_domains(generated_text)
                all_domains.append(new_domains)
                
            return all_domains
            
        except Exception as e:
            logger.error(f"Error during domain generation: {str(e)}")
            raise

    def _build_prompt(self, examples: List[str], num_to_generate: int) -> str:
        """Construct a prompt for domain generation.
        
        Args:
            examples: List of example domains
            num_to_generate: Number of domains to generate
            
        Returns:
            Formatted prompt string
        """
        shuffled_examples = random.sample(examples, len(examples))
        example_part = "\n".join([f"- {ex}" for ex in shuffled_examples])
        
        prompt = f"""
*Goal*
Build a dataset of short task instructions.

*Requirements*
1. Each instruction must be concise (≤ 4 words), specific, and unique.
2. Provide a wide thematic range; no duplicates.
3. Model‑solvable: Every instruction should describe a task that a language model can realistically address through text, reasoning, or code generation.

Examples
- "Write code"
- "Summarize analysis"
- "Compile statistics"

Phase 1 – Domains
Generate exactly {num_to_generate} real‑world domains that these tasks might address.
Criteria:
- Cover everyday life comprehensively.
- Each domain is broad, distinct, and has clear practical value.
- All tasks within the domain must be solvable (or assistable) by a language model.

Examples
{example_part}

*Strict Output Format Requirements*
- Each domain must start with a hyphen followed by a space ("- ")
- Do not number the items
- Do not include any additional text, explanations, or formatting
- Do not repeat any examples from the input
- Maintain exactly one domain per line

Output exactly {num_to_generate} domains in this format:
- Domain A
- Domain B
- Domain C
...
"""
        messages = [{"role": "user", "content": prompt}]
        
        try:
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except Exception as e:
            logger.error(f"Error applying chat template: {str(e)}")
            raise

    def _parse_generated_domains(self, text: str) -> List[str]:
        """Parse generated text to extract domains.
        
        Args:
            text: Generated text from the model
            
        Returns:
            List of extracted domains
        """
        domains = []
        lines = text.split("\n")
        
        for line in lines:
            # Handle hyphen-prefixed items (- xxx)
            if line.startswith("-"):
                domain = line[1:].strip()
            # Handle unprefixed domains
            elif line.strip() and not line.startswith("*") and ":" not in line:
                domain = line.strip()
            else:
                continue
                
            # Clean and validate
            domain = self._clean_domain(domain)
            if domain and len(domain.split()) <= 4:  # Max 4 words
                domains.append(domain)
        
        return domains

    def _clean_domain(self, domain: str) -> str:
        """Clean and normalize a domain string.
        
        Args:
            domain: Raw domain string
            
        Returns:
            Cleaned domain string
        """
        domain = domain.strip("\"'")
        domain = " ".join(domain.split())
        # Remove trailing colons if present
        if domain.endswith(":"):
            domain = domain[:-1].strip()
        return domain

    def run_batch_generation(
        self,
        output_file: str,
        iterations: int = 50,
        initial_examples: Optional[List[str]] = None,
    ) -> None:
        """Run batch generation of domains and save results.
        
        Args:
            output_file: Path to save generated domains
            iterations: Number of generation iterations
            initial_examples: Optional list of initial example domains
        """
        if initial_examples is None:
            initial_examples = [
                "Education",
                "Healthcare",
                "Finance",
                "Technology",
                "Travel",
                "Computer Science",
                "Artificial Intelligence",
                "Data Science",
                "Cybersecurity",
                "Mathematics",
            ]
        
        # Add initial examples to the final set
        self.generated_domains.update(initial_examples)
        logger.info(f"Starting with {len(initial_examples)} initial examples")
        
        # Ensure output directory exists
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Build all prompts first
        num_to_generate = 25  # Generate N domains per prompt
        prompts = []
        logger.info(f"\nBuilding {iterations} prompts with randomized example orders...")
        
        try:
            for _ in range(iterations):
                prompt = self._build_prompt(initial_examples, num_to_generate)
                prompts.append(prompt)
            
            # Generate all domains in a single batch
            logger.info("Generating all domains in a single batch...")
            all_domains_lists = self.generate_domains(prompts)
            
            # Process results
            total_new = 0
            for i, new_domains in enumerate(all_domains_lists):
                unique_new = [d for d in new_domains if d not in self.generated_domains]
                self.generated_domains.update(unique_new)
                total_new += len(unique_new)
                logger.info(f"Batch {i+1}/{iterations}: Added {len(unique_new)} new domains")
            
            # Save results
            self._save_results(output_file)
            logger.info(f"\nCompleted! Total unique domains generated: {len(self.generated_domains)}")
            
        except Exception as e:
            logger.error(f"Error during batch generation: {str(e)}")
            raise

    def _save_results(self, output_file: str) -> None:
        """Save generated domains to a file.
        
        Args:
            output_file: Path to save the results
        """
        try:
            # Sort alphabetically
            sorted_domains = sorted(self.generated_domains)
            
            with open(output_file, "w", encoding="utf-8") as f:
                for domain in sorted_domains:
                    f.write(f"{domain}\n")
                    
        except IOError as e:
            logger.error(f"Error saving results to {output_file}: {str(e)}")
            raise


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.
    
    Returns:
        Namespace object containing parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Generate diverse domains using vLLM language models."
    )
    
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        required=True,
        help="Path or name of the model to load",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="Path to save generated domains",
    )
    parser.add_argument(
        "--tp",
        type=int,
        default=4,
        help="Tensor parallel size for model distribution (default: 1)",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=4096,
        help="Maximum number of tokens to generate (default: 100)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature for generation (default: 0.7)",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=1000,
        help="Number of generation iterations (default: 50)",
    )
    parser.add_argument(
        "--initial_examples",
        type=str,
        nargs="+",
        default=[
            "Education",
            "Healthcare",
            "Finance",
            "Technology",
            "Travel",
            "Computer Science",
            "Artificial Intelligence",
            "Data Science",
            "Cybersecurity",
            "Mathematics",
        ],
        help="List of initial example domains (space-separated)",
    )
    
    return parser.parse_args()


def main() -> None:
    """Main execution function."""
    try:
        # Parse command line arguments
        args = parse_args()
        
        # Initialize generator
        generator = VLLM_DiverseGenerator(
            model_name_or_path=args.model_name_or_path,
            tp=args.tp,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
        )
        
        # Run batch generation
        generator.run_batch_generation(
            output_file=args.output_file,
            iterations=args.iterations,
            initial_examples=args.initial_examples,
        )
        
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        raise


if __name__ == "__main__":
    main()