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
    """Wrapper around vLLM for convenience."""

    def __init__(
        self,
        model_name_or_path: str,
        tp: int = 1,
        max_tokens: int = 512,
        temperature: float = 0.7,
    ) -> None:
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
        """Instantiate vLLM engine and sampling params."""
        try:
            llm_engine = LLM(
                model=self.model_name_or_path,
                tokenizer=self.model_name_or_path,
                tokenizer_mode="auto",
                tensor_parallel_size=self.tp,
                dtype="bfloat16",
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


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _load_meta_requests(input_file: str) -> List[str]:
    """Read non‑empty lines from a text file."""
    try:
        with open(input_file, "r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip()]
    except IOError as e:
        logger.error(f"Error loading input from {input_file}: {str(e)}")
        raise


def _build_persona_prompts(scenarios: List[str], count: int) -> List[str]:
    """Construct prompts that ask the model to add personas."""
    template = (
        'For the following scenario:\n"{scenario}"\n'
        'Generate {count} variations of this scenario, each featuring a distinct persona '
        'with an implicit skill level. Ensure personas come from different domains (e.g., education, healthcare, business, etc.)\n'
        'and subtly convey their skill level through job title or context (e.g., primary school teacher vs. university professor).\n'
        'Each variation should:\n'
        '1) Use a dash ("-") at the beginning\n'
        '2) Be no more than 2 sentences long\n'
        '3) Clearly embed the persona and context while preserving the original scenario theme.\n\n'
        'Example:\n'
        'Scenario: "Someone wants to schedule a video call with a colleague."\n'
        '- A university professor schedules a late‑night video call with a co‑author in another time zone to finalize a journal submission\n'
        '- A customer service representative arranges a video call with a client to resolve a technical issue\n'
        '- A primary school teacher sets up a video call with a parent to discuss the student’s recent behavioral changes\n'
        '- A freelance designer books a video call with a startup founder to pitch branding ideas\n'
        '- A hospital administrator schedules a video call with regional clinics to coordinate resource distribution\n\n'

        'Start generating now.'
    )
    return [template.format(scenario=scenario, count=count) for scenario in scenarios]


def _prepare_chat_messages(tokenizer, prompts: List[str]) -> List[str]:
    """Wrap prompts in chat template so vLLM can batch generate."""
    try:
        return [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            for prompt in prompts
        ]
    except Exception as e:
        logger.error(f"Error applying chat template: {str(e)}")
        raise


def _process_persona_outputs(
    base_scenarios: List[str],
    outputs: List,
    max_variants: int,
) -> Dict[str, List[str]]:
    """Extract dash‑prefixed lines and clean up punctuation."""
    results: Dict[str, List[str]] = {}
    for scenario, output in zip(base_scenarios, outputs):
        generated_text = output.outputs[0].text.strip()
        personas: List[str] = []
        for line in generated_text.split("\n"):
            line = line.strip()
            if line.startswith("-"):
                text = line[1:].strip()
                if text.endswith((".", "!", "?")):
                    text = text[:-1]
                if text:
                    personas.append(text)
        # deduplicate and trim
        results[scenario] = list(dict.fromkeys(personas))[:max_variants]
    return results


def _save_results_to_txt(results: Dict[str, List[str]], output_file: str) -> None:
    """Write persona variations to a text file **without** the leading dash."""
    try:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            for variations in results.values():
                for line in variations:
                    # 👉 仅写入纯文本，不再添加 "- " 前缀
                    f.write(f"{line}\n")
        logger.info(
            f"Successfully saved {sum(len(v) for v in results.values())} persona scenarios to {output_file}"
        )
    except IOError as e:
        logger.error(f"Error saving results to {output_file}: {str(e)}")
        raise


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_persona_scenarios_from_scenarios(
    vllm_engine: "VLLM_Engine",
    input_file: str,
    output_file: str,
    personas_per_scenario: int = 5,
) -> None:
    """Main helper called by CLI and by other Python code."""
    scenarios = _load_meta_requests(input_file)
    logger.info(f"Loaded {len(scenarios)} scenarios for persona expansion.")

    prompts = _build_persona_prompts(scenarios, personas_per_scenario)
    chat_messages = _prepare_chat_messages(vllm_engine.tokenizer, prompts)

    logger.info("Generating persona‑augmented scenarios …")
    outputs = vllm_engine.engine.generate(chat_messages, vllm_engine.sampling_params)

    results = _process_persona_outputs(scenarios, outputs, personas_per_scenario)
    _save_results_to_txt(results, output_file)


# ---------------------------------------------------------------------------
# CLI glue
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Expand scenarios with personas using vLLM")
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--tp", type=int, default=8)
    parser.add_argument("--max_tokens", type=int, default=8192)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--personas_per_scenario", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    vllm_engine = VLLM_Engine(
        model_name_or_path=args.model_name_or_path,
        tp=args.tp,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
    )

    generate_persona_scenarios_from_scenarios(
        vllm_engine=vllm_engine,
        input_file=args.input_file,
        output_file=args.output_file,
        personas_per_scenario=args.personas_per_scenario,
    )


if __name__ == "__main__":
    main()
