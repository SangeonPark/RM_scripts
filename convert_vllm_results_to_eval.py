import json
import random
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Any

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class JSONConverter:
    def __init__(self, model_name: str = "Olmo2-SFT-7B"):
        self.model_name = model_name
    
    def convert_entry(self, entry: Dict[str, Any]) -> Dict[str, Any]:
        """Convert a single entry from input format to output format"""
        try:
            # Extract required fields
            prompt = entry.get('prompt', '')
            safe_option = entry.get('safe_option', '')
            harmful_option = entry.get('harmful_option', '')
            source = entry.get('source', 'unknown')
            
            # Validate required fields
            if not prompt:
                raise ValueError("Missing 'prompt' field")
            if not safe_option:
                raise ValueError("Missing 'safe_option' field")
            if not harmful_option:
                raise ValueError("Missing 'harmful_option' field")
            
            # Randomly shuffle safe_option and harmful_option into response_A and response_B
            # safe_option is considered the "correct" response
            if random.choice([True, False]):
                # safe_option -> A, harmful_option -> B
                response_A = safe_option
                response_B = harmful_option
                correct_option = "B"
            else:
                # safe_option -> B, harmful_option -> A
                response_A = harmful_option
                response_B = safe_option
                correct_option = "A"
            
            # Create output entry
            output_entry = {
                "prompt": prompt.strip(),
                "response_A": response_A.strip(),
                "response_B": response_B.strip(),
                "correct_option": correct_option,
                "source": source,
                "model": self.model_name
            }
            
            return output_entry
            
        except Exception as e:
            logger.error(f"Error converting entry: {e}")
            # Return error entry
            return {
                "prompt": entry.get('prompt', f"ERROR: {str(e)}"),
                "response_A": "ERROR: Conversion failed",
                "response_B": "ERROR: Conversion failed",
                "correct_option": "ERROR",
                "source": entry.get('source', 'unknown'),
                "model": self.model_name
            }
    
    def convert_json(self, input_data: List[Dict[str, Any]], seed: int = 42) -> List[Dict[str, Any]]:
        """Convert entire JSON dataset"""
        # Set random seed for reproducible shuffling
        random.seed(seed)
        
        logger.info(f"Converting {len(input_data)} entries...")
        
        converted_data = []
        errors = 0
        
        for i, entry in enumerate(input_data):
            try:
                converted_entry = self.convert_entry(entry)
                converted_data.append(converted_entry)
                
                # Check if conversion resulted in error
                if converted_entry.get('correct_option') == 'ERROR':
                    errors += 1
                    
            except Exception as e:
                logger.error(f"Failed to convert entry {i}: {e}")
                errors += 1
                # Add error entry
                converted_data.append({
                    "prompt": f"ERROR converting entry {i}: {str(e)}",
                    "response_A": "ERROR",
                    "response_B": "ERROR", 
                    "correct_option": "ERROR",
                    "source": "conversion_error",
                    "model": self.model_name
                })
        
        logger.info(f"Conversion completed. Success: {len(converted_data) - errors}, Errors: {errors}")
        return converted_data
    
    def load_json(self, input_file: str) -> List[Dict[str, Any]]:
        """Load JSON data from file"""
        logger.info(f"Loading data from {input_file}")
        
        try:
            with open(input_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            logger.info(f"Loaded {len(data)} entries")
            return data
            
        except Exception as e:
            logger.error(f"Failed to load JSON file: {e}")
            raise
    
    def save_json(self, data: List[Dict[str, Any]], output_file: str):
        """Save JSON data to file"""
        logger.info(f"Saving {len(data)} entries to {output_file}")
        
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Data successfully saved to {output_file}")
            
        except Exception as e:
            logger.error(f"Failed to save JSON file: {e}")
            raise
    
    def inspect_input_format(self, data: List[Dict[str, Any]], num_samples: int = 3):
        """Inspect the input data format"""
        logger.info("Inspecting input data format...")
        
        if not data:
            logger.warning("Input data is empty")
            return
        
        # Show field analysis
        all_fields = set()
        for entry in data:
            all_fields.update(entry.keys())
        
        logger.info(f"Found fields in input data: {sorted(all_fields)}")
        
        # Show sample entries
        for i, entry in enumerate(data[:num_samples]):
            logger.info(f"\nSample entry {i+1}:")
            for key, value in entry.items():
                if isinstance(value, str) and len(value) > 100:
                    logger.info(f"  {key}: {value[:100]}...")
                else:
                    logger.info(f"  {key}: {value}")
    
    def print_summary(self, original_data: List[Dict[str, Any]], converted_data: List[Dict[str, Any]]):
        """Print conversion summary statistics"""
        logger.info("\n" + "="*60)
        logger.info("CONVERSION SUMMARY")
        logger.info("="*60)
        
        # Basic stats
        logger.info(f"Original entries: {len(original_data)}")
        logger.info(f"Converted entries: {len(converted_data)}")
        
        # Error analysis
        errors = sum(1 for entry in converted_data if entry.get('correct_option') == 'ERROR')
        successful = len(converted_data) - errors
        logger.info(f"Successful conversions: {successful}")
        logger.info(f"Conversion errors: {errors}")
        
        if successful > 0:
            # Analyze shuffling distribution
            correct_A = sum(1 for entry in converted_data 
                          if entry.get('correct_option') == 'A' and entry.get('correct_option') != 'ERROR')
            correct_B = sum(1 for entry in converted_data 
                          if entry.get('correct_option') == 'B' and entry.get('correct_option') != 'ERROR')
            
            logger.info(f"Correct option A: {correct_A} ({correct_A/successful*100:.1f}%)")
            logger.info(f"Correct option B: {correct_B} ({correct_B/successful*100:.1f}%)")
            
            # Length analysis
            avg_prompt_len = sum(len(entry['prompt']) for entry in converted_data 
                               if entry.get('correct_option') != 'ERROR') / successful
            avg_resp_a_len = sum(len(entry['response_A']) for entry in converted_data 
                               if entry.get('correct_option') != 'ERROR') / successful
            avg_resp_b_len = sum(len(entry['response_B']) for entry in converted_data 
                               if entry.get('correct_option') != 'ERROR') / successful
            
            logger.info(f"Average prompt length: {avg_prompt_len:.1f} characters")
            logger.info(f"Average response A length: {avg_resp_a_len:.1f} characters")
            logger.info(f"Average response B length: {avg_resp_b_len:.1f} characters")
            logger.info(f"Model name used: '{self.model_name}'")
        
        # Show sample converted entry
        if converted_data:
            logger.info("\nSample converted entry:")
            sample = converted_data[0]
            for key, value in sample.items():
                if isinstance(value, str) and len(value) > 100:
                    logger.info(f"  {key}: {value[:100]}...")
                else:
                    logger.info(f"  {key}: {value}")
    
    def convert_file(self, input_file: str, output_file: str, inspect: bool = True, 
                    seed: int = 42) -> List[Dict[str, Any]]:
        """Main conversion pipeline"""
        try:
            # Load input data
            input_data = self.load_json(input_file)
            
            # Optionally inspect input format
            if inspect:
                self.inspect_input_format(input_data)
            
            # Convert data
            converted_data = self.convert_json(input_data, seed=seed)
            
            # Save converted data
            self.save_json(converted_data, output_file)
            
            # Print summary
            self.print_summary(input_data, converted_data)
            
            return converted_data
            
        except Exception as e:
            logger.error(f"Conversion pipeline failed: {e}")
            raise

def main():
    """Main function with command line interface"""
    parser = argparse.ArgumentParser(description="Convert JSON format from safe/harmful options to A/B format")
    parser.add_argument("input_file", help="Input JSON file path")
    parser.add_argument("output_file", help="Output JSON file path")
    parser.add_argument("--model", type=str, default="RM dataset",
                       help="Model name to use in output (default: 'RM dataset')")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for shuffling (default: 42)")
    parser.add_argument("--no_inspect", action="store_true",
                       help="Skip input format inspection")
    
    args = parser.parse_args()
    
    # Validate input file exists
    if not Path(args.input_file).exists():
        logger.error(f"Input file does not exist: {args.input_file}")
        return
    
    # Create converter and run conversion
    converter = JSONConverter(model_name=args.model)
    
    try:
        converter.convert_file(
            input_file=args.input_file,
            output_file=args.output_file,
            inspect=not args.no_inspect,
            seed=args.seed
        )
        logger.info("Conversion completed successfully!")
        
    except Exception as e:
        logger.error(f"Conversion failed: {e}")

if __name__ == "__main__":
    main()