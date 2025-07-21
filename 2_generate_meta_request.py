import argparse
from pathlib import Path
import logging
from typing import List, Set, Dict
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class VLLM_DomainMetaRequestGenerator:
    """A class for generating domain-specific meta requests using vLLM language models."""

    def __init__(
        self,
        model_name_or_path: str,
        tp: int = 1,
        max_tokens: int = 512,
        temperature: float = 0.7,
    ) -> None:
        """Initialize the generator with model configuration."""
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
        self.generated_requests: Set[str] = set()

    def generate_domain_meta_requests(self, domains: List[str], requests_per_domain: int = 50) -> Dict[str, List[str]]:
        """Generate meta requests for multiple domains in a single batch."""
        try:
            # Build prompts for all domains
            prompts = [self._build_domain_prompt(domain, requests_per_domain) for domain in domains]
            
            # Batch generate all prompts at once
            logger.info(f"Generating {len(domains)} domains in a single batch...")
            outputs = self.model.generate(prompts, self.sampling_params)
            
            # Process results
            domain_requests = {}
            for domain, output in zip(domains, outputs):
                generated_text = output.outputs[0].text.strip()
                if "</think>" in generated_text:
                    generated_text = generated_text.split("</think>")[1].strip()
                
                requests = self._parse_generated_requests(generated_text)[:requests_per_domain]
                domain_requests[domain] = requests
                self.generated_requests.update(requests)
            
            return domain_requests
            
        except Exception as e:
            logger.error(f"Error during batch generation: {str(e)}")
            raise

    def _build_domain_prompt(self, domain: str, num_requests: int) -> str:
        """Construct a domain-specific prompt for meta request generation."""
        prompt_content = f"""
*Goal*  
Generate nearly {num_requests} diverse short task instructions (meta requests) specifically for the {domain} domain.

*Requirements*  
1. Each instruction must be **≤ 4 words**, specific, unique, realistic, common, and *model‑solvable*.
2. All instructions must be relevant to the {domain} domain.
3. No duplicate instructions within the output.
4. Instructions should be clear and actionable (avoid vague commands like "Do task").

*Example output for "Education" domain (Strictly follow this format and use lowercase letters)*  
- explain the math concept
- grade student essays
- create lesson plan
- suggest teaching methods  
- recommend educational apps 

Now generate nearly {num_requests} diverse meta requests specifically for the {domain} domain:
"""
        messages = [{"role": "user", "content": prompt_content}]
        
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

    def _parse_generated_requests(self, text: str) -> List[str]:
        """Parse generated text to extract meta requests."""
        requests = []
        for line in text.split("\n"):
            if line.strip().startswith("-"):
                request = line.split("-", 1)[1].strip().lower()
                if request and len(request.split()) <= 4:
                    requests.append(request)
        return requests

    def process_domains(
        self,
        domain_file: str,
        output_file: str,
        requests_per_domain: int = 30,
    ) -> None:
        """Process domains from file and save generated meta requests."""
        try:
            domains = self._load_domains(domain_file)
            logger.info(f"Loaded {len(domains)} domains from {domain_file}")
            
            domain_requests = self.generate_domain_meta_requests(domains, requests_per_domain)
            
            self._save_results(output_file, domain_requests)
            
            total_requests = sum(len(requests) for requests in domain_requests.values())
            unique_requests = len(self.generated_requests)
            logger.info(f"Generated {total_requests} requests across {len(domains)} domains")
            logger.info(f"Total unique meta requests: {unique_requests}")
            
        except Exception as e:
            logger.error(f"Error processing domains: {str(e)}")
            raise

    def _load_domains(self, domain_file: str) -> List[str]:
        """Load domains from a text file."""
        try:
            with open(domain_file, "r", encoding="utf-8") as f:
                return [line.strip() for line in f if line.strip()]
        except IOError as e:
            logger.error(f"Error loading domains: {str(e)}")
            raise

    def _save_results(self, output_file: str, domain_requests: Dict[str, List[str]]) -> None:
        """Save generated meta requests to a file after global deduplication."""
        try:
            # Collect all requests from all domains
            all_requests = []
            for requests in domain_requests.values():
                all_requests.extend(requests)
            
            # Global deduplication while preserving order
            unique_requests = list(dict.fromkeys(all_requests))
            
            # Save to file
            Path(output_file).parent.mkdir(parents=True, exist_ok=True)
            with open(output_file, "w", encoding="utf-8") as f:
                for request in unique_requests:
                    f.write(f"{request}\n")
                    
            logger.info(f"Saved {len(unique_requests)} unique meta requests to {output_file}")
            
        except IOError as e:
            logger.error(f"Error saving results: {str(e)}")
            raise


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate domain-specific meta requests using vLLM."
    )
    
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        required=True,
        help="Path or name of the model to load",
    )
    parser.add_argument(
        "--domain_file",
        type=str,
        required=True,
        help="Path to file containing domains (one per line)",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="Path to save generated meta requests",
    )
    parser.add_argument(
        "--tp",
        type=int,
        default=8,
        help="Tensor parallel size for model distribution (default: 4)",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=8192,
        help="Maximum number of tokens to generate (default: 4096)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature for generation (default: 0.6)",
    )
    parser.add_argument(
        "--requests_per_domain",
        type=int,
        default=50,
        help="Number of meta requests to generate per domain (default: 30)",
    )
    
    return parser.parse_args()


def main() -> None:
    """Main execution function."""
    try:
        args = parse_args()
        generator = VLLM_DomainMetaRequestGenerator(
            model_name_or_path=args.model_name_or_path,
            tp=args.tp,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
        )
        generator.process_domains(
            domain_file=args.domain_file,
            output_file=args.output_file,
            requests_per_domain=args.requests_per_domain,
        )
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        raise


if __name__ == "__main__":
    main()