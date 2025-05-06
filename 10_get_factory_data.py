import argparse
import json
import logging
from pathlib import Path
from typing import List, Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

class JsonToJsonlConverter:
    """Converts JSON instruction-response pairs to LLaMA-Factory compatible JSONL format."""
    
    def __init__(self, input_path: Path, output_path: Path):
        self.input_path = input_path
        self.output_path = output_path
        self._validate_paths()

    def _validate_paths(self) -> None:
        """Validate input and output paths."""
        if not self.input_path.exists():
            raise FileNotFoundError(f"Input file not found: {self.input_path}")
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

    def _read_input_json(self) -> List[Dict[str, Any]]:
        """Read and validate input JSON file."""
        try:
            with open(self.input_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if not isinstance(data, list):
                    raise ValueError("Input JSON should be an array of objects")
                return data
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON format in input file: {e}")
            raise
        except Exception as e:
            logger.error(f"Error reading input file: {e}")
            raise

    def _convert_to_llama_format(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Convert data to LLaMA-Factory format."""
        results = []
        for idx, item in enumerate(data):
            try:
                if not all(key in item for key in ("refined_instruction", "response")):
                    logger.warning(f"Missing required fields in item {idx}")
                    continue
                
                if "response" in item:
                    results.append({
                        "messages": [
                            {"role": "user", "content": item["refined_instruction"]},
                            {"role": "assistant", "content": item["response"]}
                        ]
                    })
            except Exception as e:
                logger.warning(f"Error processing item {idx}: {e}")
                continue
            
        return results

    def _write_output_jsonl(self, data: List[Dict[str, Any]]) -> None:
        """Write converted data to JSONL file."""
        try:
            with open(self.output_path, 'w', encoding='utf-8') as f:
                for item in data:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")
            logger.info(f"Successfully wrote {len(data)} records to {self.output_path}")
        except IOError as e:
            logger.error(f"Error writing output file: {e}")
            raise

    def convert(self) -> None:
        """Main conversion method."""
        try:
            input_data = self._read_input_json()
            converted_data = self._convert_to_llama_format(input_data)
            self._write_output_jsonl(converted_data)
        except Exception as e:
            logger.error(f"Conversion failed: {e}")
            raise

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Convert JSON instruction-response pairs to LLaMA-Factory JSONL format."
    )
    
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Path to input JSON file",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Path to output JSONL file",
    )
    
    return parser.parse_args()

def main() -> None:
    """Main execution function."""
    try:
        args = parse_args()
        converter = JsonToJsonlConverter(args.input, args.output)
        converter.convert()
    except Exception as e:
        logger.error(f"Script execution failed: {e}")
        raise

if __name__ == "__main__":
    main()